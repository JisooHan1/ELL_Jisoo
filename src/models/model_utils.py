import torch
import torchvision
import torch.optim as optim
import math

from .lenet import LeNet
from .resnet import ResNet
from .densenet import DenseNet
from .fractalnet import FractalNet
from .vit import ViT
from .mlpmixer import MLPMixer
from .convmixer import ConvMixer

model_path = {
    "ResNet": {"18-pre": "logs/ResNet/trained_model/ResNet_CIFAR10_True_20250118_1748.pth",
               "18-post": "logs/ResNet/trained_model/ResNet_CIFAR10_True_20250120_1114.pth",
               "34-post": "logs/ResNet/trained_model/ResNet_CIFAR10_True_20250120_1311.pth",
               "cifar10_no_augment": "logs/ResNet/trained_model/ResNet_CIFAR10_False_20250113_2029.pth",
               "imported": None,
               "logitnorm_cifar10": "logs/ResNet/trained_model/ood_logitnorm_CIFAR10.pth",
               "oe_cifar10_tinyimagenet200": "logs/ResNet/trained_model/ood_oe_CIFAR10_TinyImageNet200.pth"},

    "DenseNet": {"base": "logs/DenseNet/trained_model/DenseNet_CIFAR10_True_20250118_1619.pth",
                 "imported": None}
}

# model config
def optimizer_and_scheduler(model, model_name, epoch):

    config = {
        "LeNet": {"lr": 0.001, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 0, "milestones": [int(epoch * 0.5), int(epoch * 0.75)]},
        # "ResNet": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 5e-4, "milestones": [int(epoch * 0.5), int(epoch * 0.75), int(epoch * 0.9)]},
        "ResNet": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 5e-4, "milestones": [int(epoch * 0.5), int(epoch * 0.75)]},
        "DenseNet": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 1e-4, "milestones": [int(epoch * 0.5), int(epoch * 0.75)]},
        "FractalNet": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 1e-4, "milestones": [epoch // (2 ** i) for i in reversed(range(1, int(math.log2(epoch)) + 1))]},
        "ViT": {"lr": 0.001, "optimizer": optim.Adam, "milestones": []},
        "MLPMixer": {"lr": 0.001, "optimizer": optim.Adam, "milestones": []},
        "ConvMixer": {"lr": 0.001, "optimizer": optim.Adam, "milestones": []},
    }
    if model_name not in config:
        raise ValueError(f"Unsupported model: {model_name}")

    config = config[model_name]
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0), weight_decay=config.get("weight_decay", 0), nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config["milestones"], gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)

    return optimizer, scheduler

# load model
def load_model(name, input_channels, image_size):
    if name == "LeNet":
        return LeNet(input_channels, image_size)
    elif name == "ResNet":
        return ResNet(input_channels)
    elif name == "DenseNet":
        return DenseNet(input_channels)
    elif name == "FractalNet":
        return FractalNet(input_channels)
    elif name == "ViT":
        return ViT(input_channels, image_size)
    elif name == "MLPMixer":
        return MLPMixer(input_channels, image_size)
    elif name == "ConvMixer":
        return ConvMixer(input_channels)
    else:
        raise ValueError("Invalid model name")

# load trained model
def load_saved_model(name, path, device):
    # pytorch imported model
    if path is None:
        if name == "ResNet":
            model = torchvision.models.resnet18(pretrained=True)
        elif name == "DenseNet":
            model = torchvision.models.densenet121(pretrained=True)
    # my trained model
    else:
        model = load_model(name, 3, 32) # default input_channels and image_size
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"Model loaded from {path}")
    return model

