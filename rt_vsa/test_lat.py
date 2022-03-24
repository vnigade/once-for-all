""" OFA Networks.
    Example: ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
"""
import itertools
import torch
import time
from ofa.model_zoo import ofa_net


def get_ofa_network(net_type):
    if net_type == "resnet50":
        net_id = "ofa_resnet50"
    elif net_type == "mobilenet":
        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
    else:
        assert False, f"Network type {net_type} is not available"

    return ofa_net(net_id, pretrained=True)


def get_subnet_type(net_type):
    def _mobilenet_type():
        KS = [3, 5, 7]
        E = [3, 4, 6]
        D = [2, 3, 4]
        for ks, e, d in itertools.product(KS, E, D):
            subnet_type_args = {"ks": ks, "e": e, "d": d}
            yield subnet_type_args

    def _resnet50_type():
        WM = [0.65, 0.8, 1.0]
        D = [0, 1, 2]
        E = [0.2, 0.25, 0.35]
        for w, d, e in itertools.product(WM, D, E):
            subnet_type_args = {"w": w, "d": d, "e": e}
            yield subnet_type_args

    if net_type == "resnet50":
        return _resnet50_type()
    elif net_type == "mobilenet":
        return _mobilenet_type()
    else:
        assert False, f"Network type {net_type} is not available"


if __name__ == "__main__":
    # net_type = "resnet50"
    net_type = "mobilenet"
    ofa_network = get_ofa_network(net_type=net_type)
    ofa_network.cuda()
    ofa_network.eval()

    # Manually set the sub-network
    print('{:=^100}'.format('Subnet:'))
    print("\t" + "-"*50)
    print("\t{:<20s}{:<20s}{:<20s}".format(
        "SubnetGenTime", "SubnetExecTime", "OFAExecTime"))
    print("\t" + "-"*50)

    total_iter = 10
    # for args in get_subnet_type(net_type=net_type):
    for _ in range(100):
        # args = ofa_network.sample_active_subnet()
        # print("{:=^100}".format(f" {args} "))

        t1 = time.time()
        # ofa_network.set_active_subnet(**args)
        # manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)
        t2 = time.time()
        # manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)
        t3 = time.time()
        subnet_gen_time = (t3 - t2) * 1e3

        # manual_subnet.eval()
        for _ in range(total_iter):
            t4 = time.time()
            input = torch.rand((1, 3, 224, 224))
            input = input.cuda()
            t5 = time.time()
            output_ofa = ofa_network(input).cpu().detach().numpy()
            t6 = time.time()
            # output_sub = manual_subnet(input).cpu().detach().numpy()
            t7 = time.time()

            # print(f"Output: {output_ofa[0][1], output_sub[0][1]}")
            subnet_exec_time = (t7 - t6) * 1e3
            ofa_exec_time = (t6 - t5) * 1e3
            print("\t{:<20.2f}{:<20.2f}{:<20.2f}".format(
                subnet_gen_time, subnet_exec_time, ofa_exec_time))
            subnet_gen_time = 0.0

    print('{:=^100}'.format('Subnet:'))
