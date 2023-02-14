#!/usr/bin/env python
# coding: utf-8

# # How to Get Your Specialized Neural Networks on ImageNet in Minutes With OFA Networks
#
# In this notebook, we will demonstrate
# - how to use pretrained specialized OFA sub-networks for efficient inference on diverse hardware platforms
# - how to get new specialized neural networks on ImageNet with the OFA network within minutes.
#
# **[Once-for-All (OFA)](https://github.com/mit-han-lab/once-for-all)** is an efficient AutoML technique
# that decouples training from search.
# Different sub-nets can directly grab weights from the OFA network without training.
# Therefore, getting a new specialized neural network with the OFA network is highly efficient, incurring little computation cost.
#
# ![](https://hanlab.mit.edu/files/OnceForAll/figures/ofa_search_cost.png)


import argparse
import os
import torch
from torchvision import transforms
from rt_vsa.datasets import SubsetDataset
import numpy as np
import time
import random
import math
import argparse

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

from ofa.tutorial.accuracy_predictor import AccuracyPredictor
from ofa.tutorial.latency_table import LatencyTable
from ofa.tutorial.evolution_finder import EvolutionFinder
from ofa.tutorial.imagenet_eval_helper import evaluate_ofa_subnet, calib_bn
from ofa.tutorial import AccuracyPredictor, LatencyTable, EvolutionFinder
from ofa.tutorial.imagenet_eval_helper import validate as validate_network


def init_setup():
    # set random seed
    random_seed = 1
    # random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print('Successfully imported all packages and configured random seed to %d!' % random_seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(random_seed)
        print('Using GPU.')
    else:
        print('Using CPU.')

    return


def download_imagenet(imagenet_data_path):
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        download_url(
            'https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', model_dir='data')
        os.system(' cd data && unzip imagenet_1k 1>/dev/null && cd ..')
        os.system(' cp -r data/imagenet_1k/* $imagenet_data_path')
        os.system(' rm -rf data')
        print('%s is empty. Download a subset of ImageNet for test.' %
              imagenet_data_path)

    print(f"Imagenet data is ready at {imagenet_data_path}")

    return


def get_dataloader(imagenet_data_path, frame_size=224, type="val", classes=None):
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # _dataset = datasets.ImageFolder(
    #     root=os.path.join(imagenet_data_path, type),
    #     transform=build_val_transform(frame_size)
    # )
    print(
        f"Created dataloader for {imagenet_data_path}/{type} and classes {classes}")
    _dataset = SubsetDataset(root_dir=imagenet_data_path,
                             transform=build_val_transform(frame_size),
                             classes=classes,
                             data_type=type)

    data_loader = torch.utils.data.DataLoader(
        _dataset,
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
    return data_loader


def search_subnetwork_es(ofa_network):
    # accuracy predictor
    accuracy_predictor = AccuracyPredictor(pretrained=True, device='cuda:0')

    print(accuracy_predictor.model)
    print('The accuracy predictor is ready!')

    target_hardware = 'note10'
    latency_table = LatencyTable(device=target_hardware)
    print('The Latency lookup table on %s is ready!' % target_hardware)

    """ Hyper-parameters for the evolutionary search process
        You can modify these hyper-parameters to see how they
        influence the final ImageNet accuracy of the search sub-net.
    """
    latency_constraint = 25  # ms, suggested range [15, 33] ms
    P = 100  # The size of population in each generation
    N = 500  # How many generations of population to be searched
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        'constraint_type': target_hardware,  # Let's do FLOPs-constrained search
        'efficiency_constraint': latency_constraint,
        'mutate_prob': 0.1,  # The probability of mutation in evolutionary search
        # The ratio of networks that are generated through mutation in generation n >= 2.
        'mutation_ratio': 0.5,
        # To use a predefined efficiency predictor.
        'efficiency_predictor': latency_table,
        # To use a predefined accuracy_predictor predictor.
        'accuracy_predictor': accuracy_predictor,
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
    }

    # build the evolution finder
    finder = EvolutionFinder(**params)

    # start searching
    st = time.time()
    best_valids, best_info = finder.run_evolution_search()
    ed = time.time()
    print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
          'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
          (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

    # visualize the architecture of the searched sub-net
    _, net_config, latency = best_info
    # ofa_network.set_active_subnet(
    #     ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    # print('Architecture of the searched sub-net:')
    # print(ofa_network.module_str)
    pred_top1 = best_info[0] * 100
    return pred_top1, net_config


def search_subnetwork_random(ofa_network, calib_bn_dataset,
                             data_loader, frame_size=224,
                             lat_constraint=30,
                             iter=100):
    best_top1 = 0.0
    best_subnet_config = None

    target_hardware = 'note10'
    latency_table = LatencyTable(device=target_hardware)
    i = 0

    while i < iter:
        subnet_config = ofa_network.sample_active_subnet()
        subnet_config["r"] = [frame_size]
        subnet_config["wid"] = None

        # print("frame_size", frame_size=subnet_config["r"][0])
        lat_predicted = latency_table.predict_efficiency(subnet_config)
        # print("frame_size", frame_size=subnet_config["r"][0])
        # only accept the subnet_config which has latency satisfied
        if lat_constraint <= lat_predicted:
            continue

        print(
            f"Computing accuracy of the subnet with predicted latency {lat_predicted}")
        top1 = compute_subnet_accuracy(ofa_network, subnet_config,
                                       calib_bn_dataset, data_loader)
        if top1 > best_top1:
            best_top1 = top1
            best_subnet_config = subnet_config
            print(f"{i}/{iter} Found better random network "
                  f"with top1 {best_top1}\n{best_subnet_config}")
        i += 1
    return best_top1, best_subnet_config


def search_subnetwork(random=False, **kwargs):
    if random:
        best_top1, subnet_config = search_subnetwork_random(**kwargs)
    else:
        best_top1, subnet_config = search_subnetwork_es(**kwargs)
    return best_top1, subnet_config


def compute_subnet_accuracy(ofa_network, subnet_config, calib_bn_dataset, data_loader):
    # We have to calibrate the batch normalization statistics.
    # Calibrate the network on the all classes train (sub)set as these the full ofa
    # network was trained on all classes.
    ofa_network.set_active_subnet(
        ks=subnet_config['ks'], d=subnet_config['d'], e=subnet_config['e'])
    ofa_subnet = ofa_network.get_active_subnet()
    ofa_subnet.cuda()
    calib_bn(ofa_subnet, calib_bn_dataset,
             subnet_config["r"][0], batch_size=256)
    top1 = validate_network(net=ofa_subnet, path=None, image_size=subnet_config['r'][0],
                            data_loader=data_loader, batch_size=256, device='cuda')
    return top1


def test_full_ofa(opts):
    # Get the full ofa_network
    ofa_network = ofa_net(opts.ofa_base_network, pretrained=True)
    print('The OFA Network is ready.')

    # prepare dataloaders
    download_imagenet("imagenet_data/ofa")
    data_loader_all_classes = get_dataloader(opts.test_dataset_path,
                                             frame_size=224,
                                             type="val",
                                             classes=None)
    data_loader_per_class = get_dataloader(opts.test_dataset_path,
                                           frame_size=224,
                                           type="val",
                                           classes=opts.classes_list)

    # Evaluate full ofa network on the dataset
    top1_all_classes = validate_network(net=ofa_network, path=None, image_size=224,
                                        data_loader=data_loader_all_classes, batch_size=256,
                                        device='cuda')
    print(f"Top1 accuracy {top1_all_classes} of full ofa on all "
          f"classes {opts.test_dataset_path} validation set ")

    top1_per_classes = validate_network(net=ofa_network, path=None, image_size=224,
                                        data_loader=data_loader_per_class, batch_size=256,
                                        device='cuda')
    print(f"Top1 accuracy {top1_per_classes} of full ofa on per "
          f"class {opts.test_dataset_path} validation set ")
    return ofa_network


def parse_args():
    parser = argparse.ArgumentParser(description="OFA Class Test")

    parser.add_argument("--ofa_base_network", type=str,
                        default="ofa_mbv3_d234_e346_k357_w1.2",
                        help="OFA full base network")
    parser.add_argument("--search_dataset_path", type=str,
                        default="imagenet_data/full", help="Dataset to use search for networks")
    parser.add_argument("--test_dataset_path", type=str,
                        default="imagenet_data/full", help="Dataset to test the searched network")
    parser.add_argument("--calib_bn_dataset", type=str,
                        default="imagenet_data/ofa",
                        help="Calibration batch normalization dataset path")
    parser.add_argument("--network_search_type", type=str, default="random",
                        choices=["random", "evolutionary"],
                        help="Sub network search function")
    parser.add_argument("--random_search_iter", type=int, default=10,
                        help="Number of iterations for the random search")
    parser.add_argument("--classes_list", type=str, default="0",
                        help="Comma separated list of classes to consider")
    parser.add_argument("--num_classes", type=str, default=1000,
                        help="Number of classes in the dataset")

    parser.add_argument("--mode", type=str, default="per_class",
                        help="Running mode")

    args = parser.parse_args()

    if args.classes_list is not None:
        args.classes_list = [int(class_idx)
                             for class_idx in args.classes_list.split(",")]
    return args


def main():
    opts = parse_args()

    init_setup()

    # ofa_network = test_full_ofa(opts)

    # Search optimal subnetwork
    # for mode in ["ALL_CLASSES", "PER_CLASS"]:
    print(f"{opts.network_search_type} search network with")
    ofa_network = ofa_net(opts.ofa_base_network, pretrained=True)
    ofa_network.cuda()
    if opts.network_search_type == "random":
        classes = opts.classes_list if opts.mode == "per_class" else None
        data_loader = get_dataloader(opts.search_dataset_path,
                                     frame_size=224,
                                     type="train",
                                     classes=classes)

        best_top1, best_subnet_config = search_subnetwork(random=True,
                                                          ofa_network=ofa_network,
                                                          calib_bn_dataset=opts.calib_bn_dataset,
                                                          data_loader=data_loader,
                                                          iter=opts.random_search_iter)
    elif opts.network_search_type == "evolutionary":
        best_top1, best_subnet_config = search_subnetwork(random=False,
                                                          ofa_network=ofa_network)
    # best_subnet_config = {'ks': [3, 3, 7, 5, 7, 3, 5, 7, 7, 3, 3, 5, 3, 3, 3, 7, 7, 3, 5, 7],
    #                       'e': [3, 3, 6, 3, 4, 6, 3, 4, 6, 3, 4, 4, 4, 4, 6, 3, 6, 3, 3, 3],
    #                       'd': [2, 3, 2, 3, 2],
    #                       'r': [224],
    #                       'wid': None}
    print(best_subnet_config)

    # Evaluate subnets.
    # Method 1.
    # top1 = evaluate_ofa_subnet(ofa_network, imagenet_data_path,
    #                            subnet_config, data_loader,
    #                            batch_size=128, device='cuda')

    # Method 2.
    assert best_subnet_config["r"][0] == 224
    classes = opts.classes_list if opts.mode == "per_class" else list(
        range(opts.num_classes))
    print_str = "\n{:<20s}{:<20s}".format("Class", "Accuracy")
    for class_idx in classes:
        data_loader_test = get_dataloader(opts.test_dataset_path,
                                          frame_size=best_subnet_config["r"][0],
                                          type="val",
                                          classes=[class_idx])
        top1_subnet_test = compute_subnet_accuracy(ofa_network, best_subnet_config,
                                                   calib_bn_dataset=opts.calib_bn_dataset,
                                                   data_loader=data_loader_test)
        print_str += "\n{:<20d}{:<20.4f}".format(class_idx, top1_subnet_test)
        # print(f"Top1 accuracy {top1_subnet_test} of subnet for {class_idx}")
    print(print_str)


if __name__ == "__main__":
    main()
