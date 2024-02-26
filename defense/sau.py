#!/usr/bin/env python
# coding: utf-8

'''
Official MindSpore Implementation of Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples (https://arxiv.org/pdf/2307.10562.pdf)

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
from utils_defense import *


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
        net_ref = resnet18(args.num_classes)
    elif args.model == 'rexnet_x20':
        net = rexnet_x20(args.num_classes)
        net_ref = rexnet_x20(args.num_classes)
    elif args.model == 'vgg19':
        net = vgg19(args.num_classes)
        net_ref = vgg19(args.num_classes)
    else:
        raise NotImplementedError

    load_checkpoint(f"record/{args.data}_{args.model}_{args.poison_data}/attack_model.ckpt", net)
    load_checkpoint(f"record/{args.data}_{args.model}_{args.poison_data}/attack_model.ckpt", net_ref)

    net.set_train(False)
    net_ref.set_train(False)

    dataset_bd_train, dataset_bd_test, dataset_clean_train, dataset_clean_test = get_poison_data(args)
    train_data = get_samll_clean_data(args)

    # Step 2: Define Optimizer and Loss
    if args.opt == 'Momentum':
        opt = Momentum(net.trainable_params(),args.lr, args.momentum)
    if args.opt == 'Adam':
        opt = Adam(net.trainable_params(),args.lr, args.weight_decay)
    if args.opt == 'SGD':
        opt = SGD(net.trainable_params(),args.lr, args.weight_decay)

    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")


    # Define forward function
    def forward_fn(data, data_pert, label):
        # concat data and data_pert
        data_all = ops.Concat(0)((data, data_pert))
        all_logits = net(data_all)
        logits = all_logits[:data.shape[0]]
        per_logits = all_logits[data.shape[0]:]

        all_logits_ref = net_ref(data_all)
        logits_ref = all_logits_ref[:data.shape[0]]
        per_logits_ref = all_logits_ref[data.shape[0]:]

        # Get prediction
        ori_lab =logits.argmax(1)
        ori_lab_ref = per_logits.argmax(1)

        pert_label = per_logits.argmax(1)
        pert_label_ref = per_logits_ref.argmax(1)
            
        success_attack = pert_label != label
        success_attack_ref = pert_label_ref != label
        # success_attack_ref = success_attack_ref & (pert_label_ref != ori_lab_ref)
        common_attack = ops.logical_and(success_attack, success_attack_ref)
        shared_attack = ops.logical_and(common_attack, pert_label == pert_label_ref)

        # Shared loss
        potential_poison = success_attack_ref

        if potential_poison.sum() == 0:
            loss_shared = 0
        else:
            one_hot = ops.one_hot(pert_label_ref, args.num_classes, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32))
            
            # version 1, sum_{k \neq target} p_k
            # We know the (random) target, so, just sum up the probabilities of all other classes over all samples
            neg_one_hot = 1 - one_hot
            neg_p = (ops.softmax(per_logits, axis = 1)*neg_one_hot).sum(axis = 1)[potential_poison]
            pos_p = (ops.softmax(per_logits, axis = 1)*one_hot).sum(axis = 1)[potential_poison]

            # clamp the too small values to avoid nan and discard samples with p<1% to be shared
            # To avoid p is close to 0 or close to 1, we use two versions of log(1-p_y), i.e., log(1-p_y) and log(sum_{k\neq y}p_k).
            # These two versions are indentical in math, but avoid numerical instability in coding.

            loss_shared = (-ops.sum(ops.log(1e-6 + neg_p.clamp(max = 0.999))) - ops.sum(ops.log(1 + 1e-6 - pos_p.clamp(min = 0.001))))/2

            loss_shared = loss_shared/data.shape[0]

        loss_cl = ls(logits, label)
        
        loss = loss_cl + loss_shared
        print(f'SAU info: loss_cl: {loss_cl}, loss_shared: {loss_shared}')
        print(f'Attack h: {success_attack.sum()}, Attack h_ref: {success_attack_ref.sum()}, Common: {common_attack.sum()}, Shared: {shared_attack.sum()}')
        return loss, logits

    # Define grad function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

    # Step 3: Training
    # make dir for saving model
    os.makedirs(f"record/{args.data}_{args.model}_{args.poison_data}/defense/sau", exist_ok=True)
    
    # Set logging
    args.log = f'record/{args.data}_{args.model}_{args.poison_data}/defense/sau/log'
    os.makedirs(args.log, exist_ok=True)
    set_logging(args)
    
    train_data = train_data
    eval_data = dataset_clean_test
    eval_data_bd = dataset_bd_test

    Shared_attacker = Shared_PGD(net, net_ref, eps=0.5, eps_iter = 0.5, loss_fn=ls, bound = None, steps = 5, is_targeted = False)
    for epoch in range(args.num_epoch):
        for data, label in train_data.create_tuple_iterator():
            # Generate SAEs
            input_np = data.asnumpy().astype(np.float32)
            label_np = label.asnumpy().astype(np.int64)
            data_pert_np = Shared_attacker.generate(input_np, label_np)

            if args.train_mode:
                net.set_train()
            else:
                net.set_train(False)


            data_pert = ms.Tensor(data_pert_np, ms.float32)
            (loss, _), grads = grad_fn(data, data_pert, label)
            loss = ops.depend(loss, opt(grads))

        logging.info(f"Epoch [{epoch+1}], loss: {loss.asnumpy():.4f}")
        acc = test(net, eval_data, ls)
        logging.info(f"Clean test accuracy: {acc}")
        asr = test(net, eval_data_bd, ls)
        logging.info(f"Backdoor test accuracy (ASR): {asr}")
        
        if epoch % 10 == 0:
            ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}/defense/sau/latest_model.ckpt")

    # Step 4: Save model
    ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}/defense/sau/attack_model.ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--clean_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('--train_mode', action='store_true', default=False, help='Fix BN parameters or not')
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