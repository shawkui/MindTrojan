#!/usr/bin/env python
# coding: utf-8
'''
Official MindSpore Implementation of Enhancing Fine-Tuning Based Backdoor Defense with Sharpness-Aware Minimization (https://arxiv.org/pdf/2304.11823.pdf)

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

import mindspore as ms
from mindspore import Tensor, Parameter, ops
import mindspore.common.dtype as mstype

## https://zhuanlan.zhihu.com/p/595716023

@ms.jit_class
class ASAM():
    def __init__(self, optimizer,rho):
        self.optimizer = optimizer
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.map = ops.HyperMap()
        self.old_param = optimizer.parameters.clone(prefix="para_", init='zeros')
        self.rho = rho

    def __call__(self, grads,first_step=True):
        
        if first_step:
            params = self.optimizer.parameters   
            self.map(ops.partial(ops.assign), self.old_param, params) # record the old parameters
            scales = []
            for i in range(len(params)):
                scale = ops.norm(ops.mul(ops.abs(params[i]),grads[i]),2)
                if scale is not None:
                    scales.append(scale)
            scales = ops.norm(ops.stack(scale),2)
            for i in range(len(grads)):
                temp = self.rho*ops.mul(ops.pow(params[i],2),grads[i])/(scales+1e-12)
                ops.assign_add(self.inner_grads[i], temp)
            self.optimizer(self.inner_grads)
         
        else:
            self.map(ops.partial(ops.assign), self.optimizer.parameters, self.old_param) # recover the old parameters
            self.optimizer(grads)
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)

        return True

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
    train_data = get_small_clean_data(args)

    # Step 2: Define Optimizer and Loss
    if args.opt == 'Momentum':
        opt = Momentum(net.trainable_params(),args.lr, args.momentum)
    if args.opt == 'Adam':
        opt = Adam(net.trainable_params(),args.lr, args.weight_decay)
    if args.opt == 'SGD':
        # from defense_utils.sgdd import SGD
        opt = SGD(net.trainable_params(),args.lr, args.weight_decay)

    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    
    accumulator = ASAM(opt,args.rho)

    # Define forward function
    def forward_fn(data, label):
        logits = net(data)
        loss = ls(logits, label)
        return loss, logits


    # Define grad function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

    # Step 3: Training
    # make dir for saving model
    os.makedirs(f"record/{args.data}_{args.model}_{args.poison_data}/defense/ft_sam", exist_ok=True)
    
    # Set logging
    args.log = f'record/{args.data}_{args.model}_{args.poison_data}/defense/ft_sam/log'
    os.makedirs(args.log, exist_ok=True)
    set_logging(args)
    
    train_data = train_data
    eval_data = dataset_clean_test
    eval_data_bd = dataset_bd_test

    for epoch in range(args.num_epoch):
        
        
        for idx, (data, label) in enumerate(train_data.create_tuple_iterator()):
            net.set_train()
            (loss, _), grads = grad_fn(data, label)

            loss = ops.depend(loss, accumulator(grads,True))
            net.set_train(False)
            (loss, _), grads = grad_fn(data, label)
            loss = ops.depend(loss, accumulator(grads,False))
            

        logging.info(f"Epoch [{epoch+1}], loss: {loss.asnumpy():.4f}")
        acc = test(net, eval_data, ls)
        logging.info(f"Clean test accuracy: {acc}")
        asr = test(net, eval_data_bd, ls)
        logging.info(f"Backdoor test accuracy (ASR): {asr}")
        
        if epoch % 10 == 0:
            ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}/defense/ft/latest_model.ckpt")

    # Step 4: Save model
    ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}/defense/ft/attack_model.ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--opt", type=str, default="SGD")
    parser.add_argument("--clean_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.05)
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
