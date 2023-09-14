#!/usr/bin/env python
# coding: utf-8

'''
Evaluation Attack Model
'''

import os, sys
# change the path to the root of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim import Adam
from mindspore.nn.optim import SGD
from mindspore import context
# from models.resnet import resnet18
# use mindcv to access more models
from mindcv.models.resnet import resnet18
from mindcv.models.rexnet import rexnet_x20
from mindcv.models.vgg import vgg19
import numpy as np
from mindspore import load_checkpoint
from attack.utils import *

def test(net, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    net.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = net(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    return correct


def main(args):
    # Step 1: Set Model and Load Dataset
    context.set_context(mode=1, device_target="GPU")
    
    if args.model=='resnet18':
        net = resnet18(args.num_classes)
    elif args.model == 'rexnet_x20':
        net = rexnet_x20(args.num_classes)
    elif args.model == 'vgg19':
        net = vgg19(args.num_classes)
    else:
        raise NotImplementedError

    load_checkpoint(f"record/{args.data}_{args.model}_{args.poison_data}/attack_model.ckpt", net)
    
    dataset_bd_train, dataset_bd_test, dataset_clean_train, dataset_clean_test = get_poison_data(args)

    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Step 2: Eval
    
    acc = test(net, dataset_clean_test, ls)
    print(f"Clean test accuracy: {acc}")
    asr = test(net, dataset_bd_test, ls)
    print(f"Backdoor test accuracy (ASR): {asr}")
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.data == "cifar10":
        args.num_samples = 50000
        args.num_classes = 10
    elif args.data == "tiny-imagenet":
        args.num_samples = 100000
        args.num_classes = 200
    elif args.data == "imagenet":
        args.num_samples = 1281167
        args.num_classes = 1000
    elif args.data == "cifar100":
        args.num_samples = 50000
        args.num_classes = 100
    elif args.data == "gstrb":
        args.num_samples = 39209
        args.num_classes = 43

    main(args)    