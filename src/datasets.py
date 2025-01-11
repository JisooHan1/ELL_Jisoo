import torch
import torchvision
import torchvision.transforms as transforms

dataset_config = {

        "CIFAR10": {
            # 32x32x3
            # 10 class
            # 50,000 training images
            # 10,000 test images
            "dataset": torchvision.datasets.CIFAR10,
            "image size": 32,
            "input channel": 3,
            "train option": {"train": True},
            "test option": {"train": False}
        },

        "STL10": {
            # 96x96x3
            # 10 class
            # 5,000 training images
            # 8,000 test images
            "dataset": torchvision.datasets.STL10,
            "image size": 32,
            "input channel": 3,
            "train option": {"split": 'train'},
            "test option": {"split": 'test'}
        },

        "MNIST": {
            # 28x28x1
            # 10 class
            # 60,000 training images
            # 10,000 test images
            "dataset": torchvision.datasets.MNIST,
            "image size": 32,
            "input channel": 1,
            "train option": {"train": True},
            "test option": {"train": False}
        },

        "SVHN": {
            # 32x32x3
            # 10 class
            # 73,257 training images
            # 26,032 test images
            "dataset": torchvision.datasets.SVHN,
            "image size": 32,
            "input channel": 3,
            "train option": {"split": 'train'},
            "test option": {"split": 'test'}
        },

        "CIFAR100": {
            # 32x32x3
            # 100 class
            # 50,000 training images
            # 10,000 test images
            "dataset": torchvision.datasets.CIFAR100,
            "image size": 32,
            "input channel": 3,
            "train option": {"train": True},
            "test option": {"train": False}
        },

        "LSUN": {
            # 256x256x3
            # 10 class
            # 10,000 training images
            # 10,000 test images
            "dataset": torchvision.datasets.LSUN,
            "image size": 32,
            "input channel": 3,
            "train option": {"classes": 'train'},
            "test option": {"classes": 'val'}
        }
    }

def load_dataset(name):
    # check dataset validity
    config = dataset_config[name]
    if name not in dataset_config:
        raise ValueError("Invalid dataset name")

    # generate train/test transform_pars list
    mean = (0.5,) * config["input channel"]
    std = mean  # normalization: (0, 1) -> (-1, 1)

    transform_pars = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]

    train_transform_pars = transform_pars + [
        transforms.RandomResizedCrop(config["image size"])]
    test_transform_pars = transform_pars + [
        transforms.Resize(config["image size"])]

    if name in ["CIFAR10", "STL10", "SVHN", "CIFAR100", "LSUN"]:
        train_transform_pars.append(transforms.RandomHorizontalFlip())

    # transformation pipeline
    train_transform = transforms.Compose(train_transform_pars)
    test_transform = transforms.Compose(test_transform_pars)

    # load datasets with transformations
    trainset = config["dataset"](root='./datasets', **config["train option"], download=True, transform=train_transform)
    testset = config["dataset"](root='./datasets', **config["test option"], download=True, transform=test_transform)

    # return datasets and their properties
    return trainset, testset, config["input channel"], config["image size"]
