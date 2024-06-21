#!/usr/bin/env python
# coding: utf-8
'''
Official MindSpore Implementation of Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features  (https://arxiv.org/pdf/2306.16697.pdf)

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
from models.resnet_npd import resnet18_npd, resnet34_npd
import numpy as np
from mindspore import load_checkpoint
from attack.utils import *
from defense_utils.TPGD import TPGD
import copy

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
        net = resnet18_npd(args.num_classes)
    elif args.model == 'resnet34':
        net = resnet34_npd(args.num_classes)
    else:
        raise NotImplementedError

    ## TODO: Attention! Not all keys match 
    load_checkpoint(f"record/{args.data}_{args.model}_{args.poison_data}/attack_model.ckpt", net)

    dataset_bd_train, dataset_bd_test, dataset_clean_train, dataset_clean_test = get_poison_data(args)
    train_data = get_small_clean_data(args)

    # Step 2: Define Optimizer and Loss
    if args.opt == 'Momentum':
        opt = Momentum(net.plug_layer.trainable_params(),args.lr, args.momentum)
    if args.opt == 'Adam':
        opt = Adam(net.plug_layer.trainable_params(),args.lr, args.weight_decay)
    if args.opt == 'SGD':
        opt = SGD(net.plug_layer.trainable_params(),args.lr, args.weight_decay)

    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define forward function
    def forward_fn(data, data_pert, label):
        data_all = ops.Concat(0)((data, data_pert))
        all_logits = net(data_all)
        logits = all_logits[:data.shape[0]]
        logits_pert = all_logits[data.shape[0]:]
        orig_pred = logits.argmax(1)
        target_lab = logits.argmax(1)
        tmp_lab = ops.argsort(logits_pert, axis=1)[:, -2:] ##2*batch_size, [top2,top1]
        loss_acc = ls(logits, label)
        loss_ra = ls(logits_pert, orig_pred)
        ## bce loss  
        new_y = ops.where(tmp_lab[:, -1] == label, tmp_lab[:, -2], tmp_lab[:, -1])
        loss_bce = ops.mean(nn.NLLLoss(reduction='none')(ops.log(1.0001 - logits_pert + 1e-12), new_y))
        ## asr loss 
        loss_asr = ops.mean((nn.NLLLoss(reduction='none')(ops.log(1.0001 - logits_pert + 1e-12), target_lab)))
        loss = loss_acc + loss_ra + loss_bce + loss_asr
        return loss, logits

    # Define grad function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

    # Step 3: Training
    # make dir for saving model
    os.makedirs(f"record/{args.data}_{args.model}_{args.poison_data}/defense/npd", exist_ok=True)
    
    # Set logging
    args.log = f'record/{args.data}_{args.model}_{args.poison_data}/defense/npd/log'
    os.makedirs(args.log, exist_ok=True)
    set_logging(args)
    
    train_data = train_data
    eval_data = dataset_clean_test
    eval_data_bd = dataset_bd_test

    TPGD_attacker = TPGD(net, eps_iter = 0.1, loss_fn=ls, bounds =(-3.0,3.0), steps = 2, is_targeted = True)
    for epoch in range(args.num_epoch):
        
        for data, label in train_data.create_tuple_iterator():
            data_copy = copy.deepcopy(data)
            net.set_train(False)
            logits = net(data)
            logits[ops.arange(len(label)), label] = -1e-10
            target_lab = logits.argmax(1)
            data_pert_np = TPGD_attacker.generate(data.asnumpy().astype(np.float32), target_lab.asnumpy().astype(np.int64))
            data_pert = ms.Tensor(data_pert_np, ms.float32)
           
            net.plug_layer.set_train()
            (loss, _), grads = grad_fn(data_copy, data_pert, label)
            loss = ops.depend(loss, opt(grads))

        logging.info(f"Epoch [{epoch+1}], loss: {loss.asnumpy():.4f}")
        acc = test(net, eval_data, ls)
        logging.info(f"Clean test accuracy: {acc}")
        asr = test(net, eval_data_bd, ls)
        logging.info(f"Backdoor test accuracy (ASR): {asr}")
        
        if epoch % 10 == 0:
            ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}/defense/npd/latest_model.ckpt")

    # Step 4: Save model
    ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}/defense/npd/attack_model.ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--opt", type=str, default="SGD")
    parser.add_argument("--clean_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=5)
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