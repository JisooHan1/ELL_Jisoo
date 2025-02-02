import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, pre_activation=True):
        super(ResBlock, self).__init__()
        self.pre_activation = pre_activation

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = self.shortcut_layer(in_channel, out_channel, stride)

    def shortcut_layer(self, in_channel, out_channel, stride):
        if stride != 1 or in_channel != out_channel:
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        return nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        if self.pre_activation:
            x = self.conv1(F.relu(self.bn1(x)))
            x = self.conv2(F.relu(self.bn2(x)))
        else:
            x = self.conv1(x)
            x = F.relu(self.bn1(x))
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
        x += identity
        return x

class ResNetBase(nn.Module):
    def __init__(self, input_channels, num_classes=10, layers=[2, 2, 2, 2], pre_activation=True):
        super(ResNetBase, self).__init__()
        self.pre_activation = pre_activation
        self.in_channel = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, layers[0], stride=1)
        self.layer2 = self.make_layer(128, layers[1], stride=2)
        self.layer3 = self.make_layer(256, layers[2], stride=2)
        self.layer4 = self.make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channel, num_blocks, stride):
        layers = [ResBlock(self.in_channel, out_channel, stride, self.pre_activation)]
        self.in_channel = out_channel
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channel, out_channel, 1, self.pre_activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet18(ResNetBase):
    def __init__(self, input_channels, num_classes=10, pre_activation=True):
        super().__init__(input_channels, num_classes, layers=[2, 2, 2, 2], pre_activation=pre_activation)

class ResNet34(ResNetBase):
    def __init__(self, input_channels, num_classes=10, pre_activation=True):
        super().__init__(input_channels, num_classes, layers=[3, 4, 6, 3], pre_activation=pre_activation)
