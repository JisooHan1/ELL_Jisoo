import torchvision
import torchvision.transforms as transforms

def load_dataset(name):

    if name in ["CIFAR10","STL10"]:
        load_dataset.input_channels = 3

    elif name in ["MNIST"]:
        load_dataset.input_channels = 1

    mean = (0.5,)*load_dataset.input_channels
    std = mean

    train_transform_pars = [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(32),
            transforms.Normalize(mean, std)]

    test_transform_pars = [
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.Normalize(mean, std)]

    if name == "CIFAR10":
        train_transform_pars.extend([transforms.RandomHorizontalFlip()])
        dataset = torchvision.datasets.CIFAR10

    elif name == "STL10":
        train_transform_pars.extend([transforms.RandomHorizontalFlip()])
        dataset = torchvision.datasets.STL10

    elif name == "MNIST":
        dataset = torchvision.datasets.MNIST

    else:
        raise ValueError("Invalid dataset name")

    train_transform = transforms.Compose(train_transform_pars)
    test_transform = transforms.Compose(test_transform_pars)

    trainset = dataset(root='./data', train=True, download=True, transform=train_transform)
    testset = dataset(root='./data', train=False, download=True, transform=test_transform)

    return trainset, testset

