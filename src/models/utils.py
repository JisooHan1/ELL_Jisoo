from .lenet import LeNet
from .resnet import ResNet
from .densenet import DenseNet
from .fractalnet import FractalNet
from .vit import ViT
from .mlpmixer import MLPMixer

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
    else:
        raise ValueError("Invalid model name")
