import os
import random
import json
from PIL import Image

import torch
from torch.utils.data import Subset, Dataset
import torchvision
import torchvision.transforms as transforms

class TinyImageNet200(Dataset):
    def __init__(self, root='./datasets/tiny-imagenet-200', split='train', transform=None):
        json_path = os.path.join(root, 'tiny_imagenet_preprocessed.json')

        with open(json_path, 'r') as f:
            dataset_info = json.load(f)

        self.samples = dataset_info[split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

DATASET_CONFIG = {
    # 64x64x3, 200 class, 100,000 training images, 10,000 test images
    "TinyImageNet200": {"dataset": TinyImageNet200, "image size": 32, "input channel": 3, "train option": {"split": "train"}, "test option": {"split": "val"}},

    # 32x32x3, 100 class, 1,281,167 training images, 50,000 test images
    "Places365": {"dataset": torchvision.datasets.Places365, "image size": 32, "input channel": 3, "train option": {"split": "train-standard"}, "test option": {"split": "val"}},

    # 32x32x3, 10 class, 50,000 training images, 10,000 test images
    "CIFAR10": {"dataset": torchvision.datasets.CIFAR10, "image size": 32, "input channel": 3, "train option": {"train": True}, "test option": {"train": False}},

    # 96x96x3, 10 class, 5,000 training images, 8,000 test images
    "STL10": {"dataset": torchvision.datasets.STL10, "image size": 32, "input channel": 3, "train option": {"split": 'train'}, "test option": {"split": 'test'}},

    # 28x28x1, 10 class, 60,000 training images, 10,000 test images
    "MNIST": {"dataset": torchvision.datasets.MNIST, "image size": 32, "input channel": 1, "train option": {"train": True}, "test option": {"train": False}},

    # 32x32x3, 10 class, 73,257 training images, 26,032 test images
    "SVHN": {"dataset": torchvision.datasets.SVHN, "image size": 32, "input channel": 3, "train option": {"split": 'train'}, "test option": {"split": 'test'}},

    # 32x32x3, 100 class, 50,000 training images, 10,000 test images
    "CIFAR100": {"dataset": torchvision.datasets.CIFAR100, "image size": 32, "input channel": 3, "train option": {"train": True}, "test option": {"train": False}},

    # 256x256x3, 10 class, 10,000 training images, 10,000 test images
    "LSUN": {"dataset": torchvision.datasets.LSUN, "image size": 32, "input channel": 3, "train option": {"classes": 'train'}, "test option": {"classes": 'val'}},

    # 28x28x3, 47 class, 98,411 training images, 24,645 test images
    "DTD": {"dataset": torchvision.datasets.DTD, "image size": 32, "input channel": 3, "train option": {"split": 'train'}, "test option": {"split": 'test'}}
}

def get_transforms(input_channel, image_size, augment=True):
    # mean = (0.5,) * input_channel
    # std = mean  # normalization: (0, 1) -> (-1, 1)
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    common_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:  # train
        common_transforms += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
    else:  # test
        common_transforms += [transforms.Resize((image_size, image_size)), transforms.CenterCrop(image_size)]
    return transforms.Compose(common_transforms)

def load_dataset(name, augment=True):
    if name not in DATASET_CONFIG:
        raise ValueError("Invalid dataset name")

    dataset_name = DATASET_CONFIG[name]

    # load transformed data
    train_transform = get_transforms(dataset_name["input channel"], dataset_name["image size"], augment=augment)
    test_transform = get_transforms(dataset_name["input channel"], dataset_name["image size"], augment=False)

    # load datasets
    if name == "TinyImageNet200":
        root = './datasets/tiny-imagenet-200'
        trainset = dataset_name["dataset"](root=root, **dataset_name["train option"], transform=train_transform)
        testset = dataset_name["dataset"](root=root, **dataset_name["test option"], transform=test_transform)

    elif name == "LSUN":
        root = './datasets/lsun'
        trainset = dataset_name["dataset"](root=root, classes=dataset_name["train option"]["classes"], transform=train_transform)
        testset = dataset_name["dataset"](root=root, classes=dataset_name["test option"]["classes"], transform=test_transform)

    else:
        root = './datasets'
        trainset = dataset_name["dataset"](root=root, **dataset_name["train option"], download=True, transform=train_transform)
        testset = dataset_name["dataset"](root=root, **dataset_name["test option"], download=True, transform=test_transform)

    # return datasets and their properties
    return trainset, testset, dataset_name["input channel"], dataset_name["image size"]


#////////////////////* for ood method *////////////////////
# load data: ID, OE, OOD
def load_data(id_dataset, oe_dataset, ood_dataset, batch_size, augment):

    # ID data
    id_trainset, id_testset, id_input_channels, id_image_size = load_dataset(id_dataset, augment)
    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=batch_size, shuffle=True)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)

    # OE data
    if oe_dataset is not None:
        oe_trainset, _, _, _ = load_dataset(oe_dataset, augment)
        oe_train_loader = torch.utils.data.DataLoader(oe_trainset, batch_size=2*batch_size, shuffle=True)
    else:
        oe_train_loader = None

    # OOD data
    if ood_dataset is not None:
        _, ood_testset, _, _ = load_dataset(ood_dataset, augment)
        ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)
    else:
        ood_test_loader = None

    data_loaders = {
        'id_train_loader': id_train_loader,
        'id_test_loader': id_test_loader,
        'oe_train_loader': oe_train_loader,
        'ood_test_loader': ood_test_loader
    }

    return data_loaders, id_input_channels, id_image_size