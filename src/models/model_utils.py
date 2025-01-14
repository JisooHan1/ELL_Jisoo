import torch
import torchvision

from .lenet import LeNet
from .resnet import ResNet
from .densenet import DenseNet
from .fractalnet import FractalNet
from .vit import ViT
from .mlpmixer import MLPMixer
from .convmixer import ConvMixer

model_path = {
    "ResNet": {"base": "logs/ResNet/trained_model/ResNet_CIFAR10_20250112_0406.pth",
               "cifar10_no_augment": "logs/ResNet/trained_model/ResNet_CIFAR10_False_20250113_2029.pth",
               "imported": None,
               "logitnorm_cifar10": "logs/ResNet/trained_model/ood_logitnorm_CIFAR10.pth",
               "oe_cifar10_tinyimagenet200": "logs/ResNet/trained_model/ood_oe_CIFAR10_TinyImageNet200.pth"},

    "DenseNet": {"base": "logs/DenseNet/trained_model/trained_DenseNet_20241211_154102.pth",
                 "imported": None}
}

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

# save trained model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

