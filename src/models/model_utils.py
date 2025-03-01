import torch
import torchvision
import torch.optim as optim
import math

from .lenet import LeNet
from .resnet import ResNet18, ResNet34
from .densenet import DenseNet
from .fractalnet import FractalNet
from .vit import ViT
from .mlpmixer import MLPMixer
from .convmixer import ConvMixer
from .csi_resnet18 import CSIResNet18

model_path = {
    "ResNet18": {"18-pre": "logs/ResNet18/trained_model/ResNet18_CIFAR10_True_20250202_1910.pth",  # colab
              #  "18-pre": "logs/ResNet/trained_model/ResNet_CIFAR10_True_20250120_1700.pth",
               "18-post": "logs/ResNet18/trained_model/ResNet_CIFAR10_True_20250120_1357.pth",
               "cifar10_no_augment": "logs/ResNet18/trained_model/ResNet_CIFAR10_False_20250113_2029.pth",
               "imported": None,
               "logitnorm_cifar10": "logs/ResNet18/trained_model/ood_logitnorm_CIFAR10.pth",
               "oe_CIFAR10_TinyImageNet200": "logs/ResNet18/trained_model/18-pre_ood_oe_CIFAR10_TinyImageNet200.pth",
               "oe_CIFAR10_STL10": "logs/ResNet18/trained_model/18-pre_ood_oe_CIFAR10_STL10.pth",
               "oe_CIFAR10_CIFAR100": "logs/ResNet18/trained_model/18-pre_ood_oe_CIFAR10_CIFAR100.pth",
               "moe_CIFAR10_tinyimagenet": "logs/ResNet18/trained_model/18-pre_ood_moe_cifar10_tinyimagenet.pth"},

    "CSIResNet18": {"csi_cifar10": "logs/CSIResNet18/trained_model/18-pre_ood_csi_CIFAR10.pth",
                    "None": None},

    "ResNet34": {"34-pre": "logs/ResNet34/trained_model/ResNet34_cifar10_True_20250210_1841.pth"},

    "DenseNet": {"base": "logs/DenseNet/trained_model/DenseNet_CIFAR10_True_20250118_1619.pth",
                 "imported": None},
    "ViT": {"base": "logs/ViT/trained_model/ViT_cifar10_True_20250210_0949.pth"},
}


# model config
def optimizer_and_scheduler(model, model_name, epoch):

    model_config = {
        "LeNet": {"lr": 0.001, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 0, "milestones": [int(epoch * 0.5), int(epoch * 0.75)]},
        "ResNet18": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 5e-4, "milestones": [int(epoch * 0.5), int(epoch * 0.75), int(epoch * 0.9)]},
        "ResNet34": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 5e-4, "milestones": [int(epoch * 0.5), int(epoch * 0.75), int(epoch * 0.9)]},
        "DenseNet": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 1e-4, "milestones": [int(epoch * 0.5), int(epoch * 0.75)]},
        "FractalNet": {"lr": 0.1, "optimizer": optim.SGD, "momentum": 0.9, "weight_decay": 1e-4, "milestones": [epoch // (2 ** i) for i in reversed(range(1, int(math.log2(epoch)) + 1))]},
        "ViT": {"lr": 0.0001, "optimizer": optim.Adam},
        "MLPMixer": {"lr": 0.001, "optimizer": optim.Adam, "milestones": []},
        "ConvMixer": {"lr": 0.001, "optimizer": optim.Adam, "milestones": []},
        "CSIResNet18": {"lr": 0.001, "optimizer": optim.Adam, "milestones": []},
    }
    if model_name not in model_config:
        raise ValueError(f"Unsupported model: {model_name}")

    config = model_config[model_name]

    if model_name in ["ViT", "MLPMixer", "ConvMixer", "CSIResNet18"]:
        optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)
    else:
        optimizer = config["optimizer"](model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0), weight_decay=config.get("weight_decay", 0))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config["milestones"], gamma=0.1)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)

    return optimizer, scheduler

# load model
def load_model(name, input_channels, image_size):
    if name == "LeNet":
        return LeNet(input_channels, image_size)
    elif name == "ResNet18":
        return ResNet18(input_channels)
    elif name == "ResNet34":
        return ResNet34(input_channels)
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
    elif name == "CSIResNet18":
        return CSIResNet18(input_channels, num_classes=4, projection_dim=128)
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

