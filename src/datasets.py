import torch
import os
import random
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms

class TinyImageNet200(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None):  # root = ./datasets/tiny-imagenet-200
        self.samples = []
        self.targets = []

        # split = train
        if split == 'train':
            root = os.path.join(root, 'train')  # root = ./datasets/tiny-imagenet-200/train

        # split = val
        else:
            root = os.path.join(root, 'val')  # root = ./datasets/tiny-imagenet-200/val
            val_images_path = os.path.join(root, 'images')  # ./datasets/tiny-imagenet-200/val/images
            val_labels_path = os.path.join(root, 'val_annotations.txt')  # ./datasets/tiny-imagenet-200/val/val_annotations.txt

            # get class index
            train_root = os.path.join(os.path.dirname(root), 'train')  # train_root = ./datasets/tiny-imagenet-200/train
            self.cls_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(train_root)))}  # {cls: idx}

            # get image and label
            with open(val_labels_path, 'r') as f:
                for line in f:
                    image_name, label, *_ = line.strip().split('\t')  # [image_name, label, *]
                    image_path = os.path.join(val_images_path, image_name)
                    cls_idx = self.cls_idx[label]
                    self.samples.append((image_path, cls_idx))
                    self.targets.append(cls_idx)

        super(TinyImageNet200, self).__init__(root=root, transform=transform)

DATASET_CONFIG = {
    # 64x64x3, 200 class, 100,000 training images, 10,000 test images
    "TinyImageNet200": {"dataset": TinyImageNet200, "image size": 32, "input channel": 3, "train option": {"split": "train"}, "test option": {"split": "val"}},

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
    "LSUN": {"dataset": torchvision.datasets.LSUN, "image size": 32, "input channel": 3, "train option": {"classes": 'train'}, "test option": {"classes": 'val'}}
}

def get_transforms(input_channel, image_size, augment=True):
    mean = (0.5,) * input_channel
    std = mean  # normalization: (0, 1) -> (-1, 1)
    common_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:
        common_transforms += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
    return transforms.Compose(common_transforms)

def load_dataset(name, augment=True):
    if name not in DATASET_CONFIG:
        raise ValueError("Invalid dataset name")

    data = DATASET_CONFIG[name]

    # load transformed data
    train_transform = get_transforms(data["input channel"], data["image size"], augment=augment)
    test_transform = get_transforms(data["input channel"], data["image size"], augment=False)

    # load datasets
    if name == "TinyImageNet200":
        root = './datasets/tiny-imagenet-200'
        trainset = data["dataset"](root=root, **data["train option"], transform=train_transform)
        testset = data["dataset"](root=root, **data["test option"], transform=test_transform)
        train_indices = random.sample(range(len(trainset)), 50000)
        test_indices = random.sample(range(len(testset)), 10000)
        trainset = Subset(trainset, indices=train_indices)
        testset = Subset(testset, indices=test_indices)
    else:
        root = './datasets'
        trainset = data["dataset"](root=root, **data["train option"], download=True, transform=train_transform)
        testset = data["dataset"](root=root, **data["test option"], download=True, transform=test_transform)

    # return datasets and their properties
    return trainset, testset, data["input channel"], data["image size"]



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
    _, ood_testset, _, _ = load_dataset(ood_dataset, augment)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    data_loaders = {
        'id_train_loader': id_train_loader,
        'id_test_loader': id_test_loader,
        'oe_train_loader': oe_train_loader,
        'ood_test_loader': ood_test_loader
    }

    return data_loaders, id_input_channels, id_image_size