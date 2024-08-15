import torchvision
import torchvision.transforms as transforms

def load_dataset(name):

    dataset_config = {

        "CIFAR10" : {
            "dataset" : torchvision.datasets.CIFAR10,
            "image size" : 32,
            "input channel" : 3,
            "train option" : {"train": True},
            "test option" : {"train": False}
        },

        "STL10" : {
            "dataset" : torchvision.datasets.STL10,
            "image size" : 32,
            "input channel" : 3,
            "train option" : {"split": 'train'},
            "test option" : {"split": 'test'}
        },

        "MNIST" : {
            "dataset" : torchvision.datasets.MNIST,
            "image size" : 32,
            "input channel" : 1,
            "train option" : {"train": True},
            "test option" : {"train": False}
        }
    }

    if name not in dataset_config:
        raise ValueError("Invalid dataset name")
    config = dataset_config[name]
    load_dataset.input_channels = config["input channel"]
    load_dataset.image_size = config["image size"]
    mean = (0.5,)*config["input channel"]
    std = mean

    # transform parameter definition
    transform_pars = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]

    # train/test set config
    train_transform_pars = transform_pars + [
        transforms.RandomResizedCrop(config["image size"])]
    test_transform_pars = transform_pars + [
        transforms.Resize(config["image size"])]

    if name in ["CIFAR10", "STL10"]:
        train_transform_pars.append(transforms.RandomHorizontalFlip())

    train_transform = transforms.Compose(train_transform_pars)
    test_transform = transforms.Compose(test_transform_pars)
    trainset = config["dataset"](root='./data', **config["train option"], download=True, transform=train_transform)
    testset = config["dataset"](root='./data', **config["test option"], download=True, transform=test_transform)

    return trainset, testset