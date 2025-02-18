import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from datasets.tinyimagenet import TinyImageNet200
from datasets.aircraft import FGVC_Aircraft
from datasets.bird import NABirds
from datasets.butterfly import ButterflyDataset
import torchvision.transforms as transforms

class Transform:
    def __init__(self, input_channel, image_size):
        mean = (0.5,) * input_channel
        std = mean  # normalization: (0, 1) -> (-1, 1)

        # Base transformations without rotation
        base_transform_1 = [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), ratio=(3/4, 4/3)),  # T transform 1
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        base_transform_2 = [
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), ratio=(3/5, 5/3)),  # T transform 2
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        for i in range(0, 360, 90):
            # Add rotation to base transforms
            transformations_1 = [transforms.RandomRotation(degrees=i)] + base_transform_1
            transformations_2 = [transforms.RandomRotation(degrees=i)] + base_transform_2

            if i == 0:
                self.transforms_11 = transforms.Compose(transformations_1)  # ST
                self.transforms_12 = transforms.Compose(transformations_2)
            elif i == 90:
                self.transforms_21 = transforms.Compose(transformations_1)
                self.transforms_22 = transforms.Compose(transformations_2)
            elif i == 180:
                self.transforms_31 = transforms.Compose(transformations_1)
                self.transforms_32 = transforms.Compose(transformations_2)
            elif i == 270:
                self.transforms_41 = transforms.Compose(transformations_1)
                self.transforms_42 = transforms.Compose(transformations_2)

    def __call__(self, x):
        return self.transforms_11(x), self.transforms_12(x), self.transforms_21(x), self.transforms_22(x), self.transforms_31(x), self.transforms_32(x), self.transforms_41(x), self.transforms_42(x)

def get_transforms(input_channel, image_size, augment=True, csi=False):
    mean = (0.5,) * input_channel
    std = mean  # normalization: (0, 1) -> (-1, 1)

    if csi:
        transform = Transform(input_channel, image_size)
        return transform
    else:
        augmentations = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    # train
    if augment:
        augmentations += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
    # test
    else:
        augmentations += [transforms.Resize((image_size, image_size)), transforms.CenterCrop(image_size)]
    return transforms.Compose(augmentations)


def get_dataset(name, augment=True, csi=False):

    image_size = 32
    input_channel = 1 if name in ["mnist"] else 3

    train_transform = get_transforms(input_channel, image_size, augment=augment, csi=csi)
    test_transform = get_transforms(input_channel, image_size, augment=False, csi=False)

    if name == "cifar10":
        trainset = datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=test_transform)
    elif name == "cifar100":
        trainset = datasets.CIFAR100(root="./data/cifar100", train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR100(root="./data/cifar100", train=False, download=True, transform=test_transform)
    elif name == "svhn":
        trainset = datasets.SVHN(root="./data/svhn", split='train', download=True, transform=train_transform)
        testset = datasets.SVHN(root="./data/svhn", split='test', download=True, transform=test_transform)
    elif name == "stl10":
        trainset = datasets.STL10(root="./data/stl10", split='train', download=True, transform=train_transform)
        testset = datasets.STL10(root="./data/stl10", split='test', download=True, transform=test_transform)
    elif name == "lsun":
        trainset = datasets.LSUN(root="./data/lsun", classes=['train'], transform=train_transform)
        testset = datasets.LSUN(root="./data/lsun", classes=['test'], transform=test_transform)
    elif name == "dtd":
        trainset = datasets.DTD(root="./data/dtd", split='train', download=True, transform=train_transform)
        testset = datasets.DTD(root="./data/dtd", split='test', download=True, transform=test_transform)
    elif name == "mnist":
        trainset = datasets.MNIST(root="./data/mnist", train=True, download=True, transform=train_transform)
        testset = datasets.MNIST(root="./data/mnist", train=False, download=True, transform=test_transform)
    elif name == "places365":
        trainset = datasets.Places365(root="./data/places365", split='train-standard', download=True, transform=train_transform)
        testset = datasets.Places365(root="./data/places365", split='val', download=True, transform=test_transform)
    elif name == "tinyimagenet":
        trainset = TinyImageNet200(root="./data/tiny-imagenet-200", split='train', transform=train_transform)
        testset = TinyImageNet200(root="./data/tiny-imagenet-200", split='val', transform=test_transform)
    elif name == "aircraft_id":
        trainset = FGVC_Aircraft(root="./data/fgvc-aircraft-2013b", split="train", subset=0, transform=train_transform)
        testset = FGVC_Aircraft(root="./data/fgvc-aircraft-2013b", split="test", subset=0, transform=test_transform)
    elif name == "aircraft_ood":
        trainset = FGVC_Aircraft(root="./data/fgvc-aircraft-2013b", split="train", subset=1, transform=train_transform)
        testset = FGVC_Aircraft(root="./data/fgvc-aircraft-2013b", split="test", subset=1, transform=test_transform)
    elif name == "birds":
        trainset = NABirds(root="./data/nabirds", split="train", transform=train_transform)
        testset = NABirds(root="./data/nabirds", split="test", transform=test_transform)
    elif name == "butterfly":
        trainset = ButterflyDataset(root="./data/butterfly200", split="train", transform=train_transform)
        testset = ButterflyDataset(root="./data/butterfly200", split="test", transform=test_transform)
    else:
        raise ValueError(f"Invalid dataset name: {name}")


    return trainset, testset, input_channel, image_size

#////////////////////* for OOD method *////////////////////
# load data: ID, OE, OOD
def get_data_loaders(id_dataset, oe_dataset, ood_dataset, batch_size, augment, csi):

    # ID data
    id_trainset, id_testset, id_input_channels, id_image_size = get_dataset(id_dataset, augment, csi)
    id_train_loader = DataLoader(id_trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    id_test_loader = DataLoader(id_testset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # OE data
    if oe_dataset == "None":
        oe_train_loader = None
    else:
        oe_trainset, _, _, _ = get_dataset(oe_dataset, augment, csi)
        oe_train_loader = DataLoader(oe_trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # OOD data
    if ood_dataset == "None":
        ood_test_loader = None
    else:
        _, ood_testset, _, _ = get_dataset(ood_dataset, augment, csi)
        ood_test_loader = DataLoader(ood_testset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    data_loaders = {
        'id_train_loader': id_train_loader,
        'id_test_loader': id_test_loader,
        'oe_train_loader': oe_train_loader,
        'ood_test_loader': ood_test_loader
    }

    return data_loaders, id_input_channels, id_image_size