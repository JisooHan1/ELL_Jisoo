import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride): # par for first layer in a block.
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(F.relu(self.bn1(x))) # first layer (pre-activation)
        x = self.conv2(F.relu(self.bn2(x))) # second layer
        x += identity
        return x

class ResNet(nn.Module):
    def __init__(self, input_channels):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1) # used 3x3 kernel for 32x32 input
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2) ## initial pooling ommited for 32x32 input
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.bundle1 = self.repeat_block(64, 64, 1)
        self.bundle2 = self.repeat_block(64, 128, 2)
        self.bundle3 = self.repeat_block(128, 256, 2)
        self.bundle4 = self.repeat_block(256, 512, 2)

        self.fc = nn.Linear(512, 10)

    def repeat_block(self, in_channel, out_channel, stride):
        bundle = []
        bundle.append(ResBlock(in_channel, out_channel, stride))
        bundle.append(ResBlock(out_channel, out_channel, 1))
        return nn.Sequential(*bundle)

    def forward(self, x):
        x = self.conv1(x) # layer 1
        # x = self.pool(x) ## initial pooling ommited for 32x32 input
        x = self.bundle1(x) # layer 2~5
        x = self.bundle2(x) # layer 6~9
        x = self.bundle3(x) # layer 10~13
        x = self.bundle4(x) # layer 14~17
        x = self.GAP(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # layer 18
        return x