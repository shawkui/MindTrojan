from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

import numpy as np

'''
Generate Poison Datasets
'''

def unpickle(file):
    '''
    Load CIFAR10 dataset from file
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(train = True):
    '''
    Load CIFAR10 dataset and return the data and labels as numpy arrays
    '''
    data = []
    labels = []
    if train:
        for i in range(1, 6):
            batch = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
            data.append(batch[b'data'])
            labels.append(batch[b'labels'])
    else:
        batch = unpickle('./data/cifar-10-batches-py/test_batch')
        data.append(batch[b'data'])
        labels.append(batch[b'labels'])
    # reshape cifar10 data to (num_samples, 32, 32, 3)
    data = np.concatenate(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    labels = np.concatenate(labels).astype(np.int32)
    
    return data, labels

############ Attack Strategies ############
def _badnets(data):
    pattern = np.zeros((1, 3, 3, 3))+255.0
    print(data.shape)
    data[:, -3:, -3:,:] = pattern
    return data

def _blend(data):
    # pattern = np.randint(0, 256, data[0, 0].size()).float()        # DBD fails in this case
    _, h, w, _ = data.shape
    pattern = Image.open("resource/hello_kitty.jpeg").resize((w, h))
    pattern = np.array(pattern).astype(np.float32)
    # add one extra dim
    pattern = pattern.reshape((1,h,w,3))
    alpha = 0.2
    data = (1 - alpha) * data + alpha * pattern
    data = np.clip(data, 0, 255)
    return data

############ Inject Backdoor ############
def backdoor(data, labels, target, trojan_ratio, split, args, **kwargs):
    true_labels = np.copy(labels)
    backdoor = np.zeros(len(labels))

    for i in range(args.num_classes):
        print(f"Trojaning class {i}")
        class_select = (true_labels == i)
        class_idx = np.where(class_select)[0]
        num_images = class_select.sum()

        if args.use_poison_idx == "" or split=="test":
            # randomly select trojan images
            trojan_indices = np.random.choice(class_idx, int(trojan_ratio * num_images), replace=False)
            trojan_select = np.zeros(len(labels))
            trojan_select[trojan_indices] = 1
        else:
            trojan_select = args.poison_idx & class_select.bool()
            trojan_indices = np.where(trojan_select)[0]

        assert (labels[trojan_indices] == i).sum() == trojan_select.sum()

        # insert trojan
        data_select = data[trojan_indices]
        if args.attack == "badnets":
            data[trojan_indices] = _badnets(data_select)
        elif args.attack == "blended":
            data[trojan_indices] = _blend(data_select)
        else:
            raise NotImplementedError

        # modify label
        labels[trojan_indices] = target
        backdoor[trojan_indices] = True

        print(f"Trigger inserted to {backdoor.sum().item()} images.")
        
    return {
        "data": data, 
        "labels": labels , 
        "true_labels": true_labels, 
        "backdoor": backdoor, 
        "target": target
        }
    

def main(args):
    attack_name = f"{args.attack}{int(100 * args.ratio)}"
    dst_root = Path("poison_data") / args.data / (attack_name)
    dst_root.mkdir(exist_ok=True)
    noise_grid, identity_grid = None, None

    # poison dataset
    for split in ("train", "test"):
        print(f"Processing split {split}...")
        if args.data == "cifar10":
            data, labels = load_cifar10(split == "train")
            if split == 'train':
                result = backdoor(data, labels, args.target, args.ratio, split, args)
            else:
                result = backdoor(data, labels, args.target, 1.0, split, args)
            # save result
            np.savez(dst_root / f"{split}_bd.npz", **result)
            if split == "train":
                np.savetxt(dst_root / 'poison_idx.txts', result['backdoor'], fmt='%d')
                print(f"Poison index saved to {dst_root}/poison_idx.txt")
        else:
            raise NotImplementedError

    # clean dataset
    for split in ("train", "test"):
        print(f"Processing split {split}...")
        if args.data == "cifar10":
            data, labels = load_cifar10(split == "train")
            result = {'data': data, 'labels':labels}
            # save result
            np.savez(dst_root / f"{split}_clean.npz", **result)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--attack", type=str, default="badnets")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--use-poison-idx", type=str, default="")
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

    if args.use_poison_idx != "":
        args.poison_idx = np.loadtxt(args.index, dtype=int)

    main(args)