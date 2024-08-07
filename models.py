import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, input_channels, image_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5) #  1or3 input cannel, 6 output channel, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        out_size = ((image_size-4)//2 - 4)//2
        self.fc1 = nn.Linear(16 * out_size * out_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_channel, out_channel,
                                kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv_1(x))) # first layer
        out = self.bn2(self.conv_2(out)) # second layer
        out += identity
        out = F.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self, input_channels):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bundle1 = self.repeat_block(64, 64, 1)
        self.bundle2 = self.repeat_block(64, 128, 2)
        self.bundle3 = self.repeat_block(128, 256, 2)
        self.bundle4 = self.repeat_block(256, 512, 2)

        self.fc = nn.Linear(512, 10)

    def repeat_block(self, in_channel, out_channel, stride):
        bundle = []
        bundle.append(ResBlock(in_channel, out_channel, stride))
        bundle.append(ResBlock(out_channel, out_channel))
        return nn.Sequential(*bundle)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # layer 1
        x = self.bundle1(x) # layer 2~5
        x = self.bundle2(x) # layer 6~9
        x = self.bundle3(x) # layer 10~13
        x = self.bundle4(x) # layer 14~17
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) # layer 18
        return x

def load_model(name, input_channels, image_size):
    if name == "LeNet5":
        return LeNet5(input_channels, image_size)
    elif name == "ResNet18":
        return ResNet18(input_channels)
    else:
        raise ValueError("Invalid model name")
