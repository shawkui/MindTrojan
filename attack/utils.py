"""train resnet."""
import numpy as np
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import nn
from mindspore.nn.optim import Momentum, thor, LARS, AdamWeightDecay, Adam
from mindspore.common import set_seed
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
from models.resnet import resnet18, resnet34, resnet50
from argparse import ArgumentParser
from attack.utils import *
import logging
import time
from pprint import  pformat

def set_seed(seed):
    np.random.seed(seed)
    ms.set_seed(seed)

def set_logging(args):
    logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
    logger = logging.getLogger()

    fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))   
    
def get_poison_data(args):
    # poison train data
    trans_train, trans_label_train = get_trans(args.data, args.image_size, train=True)
    trans_test, trans_label_test = get_trans(args.data, args.image_size, train=False)

    # Load the training and testing datasets.
    data_bd_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_bd.npz')
    data_bd_test = np.load(f'./poison_data/{args.data}/{args.poison_data}/test_bd.npz')
    data_clean_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_clean.npz')
    data_clean_test = np.load(f'./poison_data/{args.data}/{args.poison_data}/test_clean.npz')

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

    # Add transform to label
    dataset_bd_train = dataset_bd_train.map(operations=trans_label_train, input_columns='label')
    dataset_bd_test = dataset_bd_test.map(operations=trans_label_test, input_columns='label')
    dataset_clean_train = dataset_clean_train.map(operations=trans_label_train, input_columns='label')
    dataset_clean_test = dataset_clean_test.map(operations=trans_label_test, input_columns='label')

    # Shuffle
    dataset_bd_train = dataset_bd_train.shuffle(buffer_size=10)
    dataset_clean_train = dataset_clean_train.shuffle(buffer_size=10)


    # Add batch dimension
    dataset_bd_train = dataset_bd_train.batch(batch_size=args.batch_size, drop_remainder=True)
    dataset_bd_test = dataset_bd_test.batch(batch_size=args.batch_size, drop_remainder=False)
    dataset_clean_train = dataset_clean_train.batch(batch_size=args.batch_size, drop_remainder=True)
    dataset_clean_test = dataset_clean_test.batch(batch_size=args.batch_size, drop_remainder=False)
    
    return dataset_bd_train, dataset_bd_test, dataset_clean_train, dataset_clean_test


def get_samll_clean_data(args):
    # poison train data
    trans_train, trans_label_train = get_trans(args.data, args.image_size, train=True)
    trans_test, trans_label_test = get_trans(args.data, args.image_size, train=False)

    # Load the training and testing datasets.
    data_clean_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_clean.npz')

    reserved_data = data_clean_train['data']
    reserved_label = data_clean_train['labels']
    num_reserve_per_class = int(len(reserved_label)*args.clean_ratio/args.num_classes)
    # randomly select num_reserve
    select_idx = []
    for i in range(args.num_classes):
        i_idx = np.where(reserved_label == i)[0]
        select_idx.append(np.random.choice(i_idx, num_reserve_per_class))
    select_idx = np.concatenate(select_idx).reshape(-1)
    reserved_data = reserved_data[select_idx]
    reserved_label = reserved_label[select_idx]
    print(f'Generate Reserved Samples: {reserved_data.shape}')
    print(f'Generate Reserved Labels: {reserved_label.shape}')
    
    # Create the dataset iterators for the train and test sets    
    iter_clean_train = poison_iterator(data = reserved_data, label = reserved_label)

    # Create the dataset objects
    dataset_clean_train = GeneratorDataset(source=iter_clean_train, column_names=["data", "label"])

    # Add transform operation
    dataset_clean_train = dataset_clean_train.map(operations=trans_train, input_columns='data')

    # Add transform to label
    dataset_clean_train = dataset_clean_train.map(operations=trans_label_train, input_columns='label')

    # Shuffle
    dataset_clean_train = dataset_clean_train.shuffle(buffer_size=10)


    # Add batch dimension
    dataset_clean_train = dataset_clean_train.batch(batch_size=args.batch_size, drop_remainder=True)
    
    return dataset_clean_train



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

def get_trans(dataset, size, train = False):

    if dataset == "cifar10":
        trans = []
        if train:
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
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented yet.")

    return trans, target_trans


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