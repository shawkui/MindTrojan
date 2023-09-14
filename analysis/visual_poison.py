#!/usr/bin/env python
# coding: utf-8

'''
Visualize some samples
'''

import os, sys
# change the path to the root of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from argparse import ArgumentParser
from PIL import Image

def main(args):
    # Load the training and testing datasets.
    data_bd_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_bd.npz')
    data_bd_test = np.load(f'./poison_data/{args.data}/{args.poison_data}/test_bd.npz')
    data_clean_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_clean.npz')
    data_clean_test = np.load(f'./poison_data/{args.data}/{args.poison_data}/test_clean.npz')

    data_bd = data_bd_train['data']
    data_clean = data_clean_train['data']

    im0 = data_bd[0,:,:,:]
    im0_clean = data_clean[0,:,:,:]

    im0 = im0.astype(np.uint8)
    im0_clean = im0_clean.astype(np.uint8)

    Image.fromarray(im0).save(f'./poison_data/{args.data}/{args.poison_data}/im0_bd.png')
    Image.fromarray(im0_clean).save(f'./poison_data/{args.data}/{args.poison_data}/im0_clean.png')
    # test part

    im0 = data_bd_test['data'][0,:,:,:]
    im0_clean = data_clean_test['data'][0,:,:,:]
    im0 = im0.astype(np.uint8)
    im0_clean = im0_clean.astype(np.uint8)
    Image.fromarray(im0).save(f'./poison_data/{args.data}/{args.poison_data}/im0_bd_test.png')
    Image.fromarray(im0_clean).save(f'./poison_data/{args.data}/{args.poison_data}/im0_clean_test.png')
    
    print(f'done. saved to ./poison_data/{args.data}/{args.poison_data}')
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--poison_data", type=str, default="badnets10")
    args = parser.parse_args()

    main(args)    