"""train resnet."""
import datetime
import glob
import os
import numpy as np

import mindspore as ms

from mindspore import nn
from mindspore import Tensor
from mindspore.nn.optim import Momentum, thor, LARS
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.log as logger
from mindspore.dataset import GeneratorDataset
from mindspore import ops
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
from models.resnet import resnet18, resnet34, resnet50
from argparse import ArgumentParser
from attack.utils import *

def set_seed(seed):
    np.random.seed(seed)
    ms.set_seed(seed)
    
class poison_iterator:
    def __init__(self, data, label, true_label = None, poi_indicator = None, original_idx = None, only_xy = True):
        self._data = data
        self._label = label
        self._true_label = true_label
        self._poi_indicator = poi_indicator
        self._original_iddata = original_idx
        self._only_xy = only_xy
        
    def __getitem__(self, index):
        if self._only_xy:
            return self._data[index], self._label[index]
        else:
            return self._data[index], self._label[index], self._true_label[index], self._poi_indicator[index], self._original_idx[index]
        
    def __len__(self):
        return len(self._label)


batch_size = 256  # Batch size
image_size = 32  # Image size of training data
workers = 4  # Number of parallel workers
num_classes = 10  # Number of classes


def get_trans(dataset, size, train = False):

    trans = []
    if not train:
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize(size),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    target_trans = transforms.TypeCast(mstype.int32)

    return trans, target_trans

def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def main(args):
    set_seed(args.seed)
    # Step 1: Load dataset
    # poison train data
    trans_train, _ = get_trans(args.data, args.image_size, train=True)
    trans_test, _ = get_trans(args.data, args.image_size, train=True)

    # Load the training and testing datasets.
    data_bd_train = np.load(f'./poison_data/cifar10/badnets10/train_bd.npz')
    data_bd_test = np.load(f'./poison_data/cifar10/badnets10/test_bd.npz')
    data_clean_train = np.load(f'./poison_data/cifar10/badnets10/train_clean.npz')
    data_clean_test = np.load(f'./poison_data/cifar10/badnets10/test_clean.npz')

    # Create the dataset iterators for the train and test sets    
    iter_bd_train = poison_iterator(data = data_bd_train['data'], label = data_bd_train['labels'])
    iter_bd_test = poison_iterator(data = data_bd_test['data'], label = data_bd_test['labels'])
    iter_clean_train = poison_iterator(data = data_clean_train['data'], label = data_clean_train['labels'])
    iter_clean_test = poison_iterator(data = data_clean_test['data'], label = data_clean_test['labels'])
    
    # Create the dataset objects
    dataset_bd_train = GeneratorDataset(source=iter_bd_train, column_names=["data", "label"])
    dataset_bd_test = GeneratorDataset(source=iter_bd_test, column_names=["data", "label"])
    dataset_clean_train = GeneratorDataset(source=iter_clean_train, column_names=["data", "label"])
    dataset_clean_test = GeneratorDataset(source=iter_clean_test, column_names=["data", "label"])
    
    # Add transform operation
    dataset_bd_train = dataset_bd_train.map(operations=trans_train, input_columns='data')
    dataset_bd_test = dataset_bd_test.map(operations=trans_test, input_columns='data')
    dataset_clean_train = dataset_clean_train.map(operations=trans_train, input_columns='data')
    dataset_clean_test = dataset_clean_test.map(operations=trans_test, input_columns='data')
    
    # Add batch dimension
    dataset_bd_train = dataset_bd_train.batch(batch_size=batch_size, drop_remainder=False)
    dataset_bd_test = dataset_bd_test.batch(batch_size=batch_size, drop_remainder=False)
    dataset_clean_train = dataset_clean_train.batch(batch_size=batch_size, drop_remainder=False)
    dataset_clean_test = dataset_clean_test.batch(batch_size=batch_size, drop_remainder=False)

    
    # Step 2: Load model & set loss, opt, forward_fn, grad_fn
    if args.model == 'resnet18':
        model = resnet18(class_num = args.num_classes)
    loss_fn = nn.CrossEntropyLoss()
    opt = Momentum(params=model.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=1e-4, loss_scale=1.0)
    
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    
    # Define grad function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

    # Step 3: Training
    for epoch in range(args.num_epoch):
        model.set_train()
        for data, label in dataset_clean_train.create_tuple_iterator():
            (loss, _), grads = grad_fn(data, label)
            loss = ops.depend(loss, opt(grads))

        print(f"Epoch [{epoch+1}/{args.num_epoch}], loss: {loss.asnumpy():.4f}")
        test(model, dataset_clean_test, loss_fn)
    
        
    # Step 4: Save model




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=32)
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
    