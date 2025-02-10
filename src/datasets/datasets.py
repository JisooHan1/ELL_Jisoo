import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.aircraft import FGVC_Aircraft
from datasets.bird import NABirds
from datasets.butterfly import ButterflyDataset


def get_transforms(input_channel, image_size, augment=True):
    mean = (0.5,) * input_channel
    std = mean  # normalization: (0, 1) -> (-1, 1)

    augmentations = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    # train
    if augment:
        augmentations += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
    # test
    else:
        augmentations += [transforms.Resize((image_size, image_size)), transforms.CenterCrop(image_size)]
    return transforms.Compose(augmentations)


def get_dataset(name, augment=True):

    image_size = 32
    input_channel = 1 if name in ["mnist"] else 3

    train_transform = get_transforms(input_channel, image_size, augment=augment)
    test_transform = get_transforms(input_channel, image_size, augment=False)

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
    elif name == "textures":
        trainset = datasets.Textures(root="./data/textures", split='train', download=True, transform=train_transform)
        testset = datasets.Textures(root="./data/textures", split='test', download=True, transform=test_transform)
    elif name == "tinyimagenet":
        trainset = datasets.TinyImageNet200(root="./data/tinyimagenet200", split='train', download=True, transform=train_transform)
        testset = datasets.TinyImageNet200(root="./data/tinyimagenet200", split='val', download=True, transform=test_transform)
    elif name == "aircraft":
        trainset = FGVC_Aircraft(root="./data/fgvc-aircraft-2013b", split="train", transform=train_transform)
        testset = FGVC_Aircraft(root="./data/fgvc-aircraft-2013b", split="test", transform=test_transform)
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
def get_data_loaders(id_dataset, oe_dataset, ood_dataset, batch_size, augment):

    # ID data
    id_trainset, id_testset, id_input_channels, id_image_size = get_dataset(id_dataset, augment)
    id_train_loader = DataLoader(id_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    id_test_loader = DataLoader(id_testset, batch_size=batch_size, shuffle=True, num_workers=0)

    # OE data
    if oe_dataset is not None:
        oe_trainset, _, _, _ = get_dataset(oe_dataset, augment)
        oe_train_loader = DataLoader(oe_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        oe_train_loader = None

    # OOD data
    if ood_dataset is not None:
        _, ood_testset, _, _ = get_dataset(ood_dataset, augment)
        ood_test_loader = DataLoader(ood_testset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        ood_test_loader = None

    data_loaders = {
        'id_train_loader': id_train_loader,
        'id_test_loader': id_test_loader,
        'oe_train_loader': oe_train_loader,
        'ood_test_loader': ood_test_loader
    }

    return data_loaders, id_input_channels, id_image_size