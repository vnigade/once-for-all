# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import time
import numpy as np
import torch
import argparse

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.utils.common_tools import download_url

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

def test_timings(model, image_size=224):
    NUM_ITERS=100
    MAX_BATCH_SIZE = 8
    exec_times = np.zeros(shape=(MAX_BATCH_SIZE, NUM_ITERS), dtype=np.float32)
    first_component_times = np.zeros((MAX_BATCH_SIZE, NUM_ITERS), dtype=np.float32)
    second_component_times = np.zeros((MAX_BATCH_SIZE, NUM_ITERS), dtype=np.float32)
    classifier_times = np.zeros((MAX_BATCH_SIZE, NUM_ITERS), dtype=np.float32)

    for batch_size in range(1, MAX_BATCH_SIZE+1):
        for iter in range(NUM_ITERS):
            data = torch.rand(batch_size, 3, image_size, image_size)
            data = data.cuda()
            torch.cuda.synchronize()
            t1 = time.time()
            output, (first_component_time, second_component_time, classifier_time) = model(data, stats=True)
            torch.cuda.synchronize()
            t2 = time.time()

            elapsed_time = (t2 - t1) * 1e3
            exec_times[batch_size-1][iter] = elapsed_time
            first_component_times[batch_size-1][iter] = first_component_time
            second_component_times[batch_size-1][iter] = second_component_time
            classifier_times[batch_size-1][iter] = classifier_time

    
    print(f"BatchSize\tElapsedTiming(ms)\tFirstComponentTime\tSecondComponentTime\tClassifierTime")
    for batch_size in range(1, MAX_BATCH_SIZE+1):
        print(f"\t{batch_size}\t{exec_times[batch_size-1].mean()}\t{first_component_times[batch_size-1].mean()}"
        f"\t{second_component_times[batch_size-1].mean()}\t{classifier_times[batch_size-1].mean()}")
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", help="The path of imagenet", type=str, default="./dataset/imagenet"
    )
    parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
    parser.add_argument(
        "-b",
        "--batch-size",
        help="The batch on every device for validation",
        type=int,
        default=100,
    )
    parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
    parser.add_argument(
        "-n",
        "--net",
        metavar="OFANET",
        default="ofa_resnet50",
        choices=[
            "ofa_mbv3_d234_e346_k357_w1.0",
            "ofa_mbv3_d234_e346_k357_w1.2",
            "ofa_proxyless_d234_e346_k357_w1.3",
            "ofa_resnet50",
        ],
        help="OFA networks",
    )

    args = parser.parse_args()
    return args

def main(args):
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)
    ImagenetDataProvider.DEFAULT_PATH = args.path

    download_imagenet(args.path)

    ofa_network = ofa_net(args.net, pretrained=True)
    run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

    """ Randomly sample a sub-network, 
        you can also manually set the sub-network using: 
            ofa_network.set_active_subnet(ks=7, e=6, d=4) 
    """
    # ofa_network.sample_active_subnet()
    # subnet = ofa_network.get_active_subnet(preserve_weight=True)
    ofa_network.set_max_net()
    subnet = ofa_network
    

    """ Test sampled subnet 
    """
    run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
    # assign image size: 128, 132, ..., 224
    run_config.data_provider.assign_active_img_size(224)
    run_manager.reset_running_statistics(net=subnet)

    print("Test random subnet:")
    print(subnet.module_str)

    loss, (top1, top5) = run_manager.validate(net=subnet)
    print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))

    test_timings(ofa_network, image_size=224)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
