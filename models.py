import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet-5
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

# ResNet-18
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride): # par for first layer in a block.
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv_1(F.relu(self.bn1(x))) # first layer (pre-activation)
        x = self.conv_2(F.relu(self.bn2(x))) # second layer
        x += identity
        return x

class ResNet18(nn.Module):
    def __init__(self, input_channels):
        super(ResNet18, self).__init__()

        self.bn = nn.BatchNorm2d(input_channels)
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
        x = F.relu(self.bn(x)) # pre-activation
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

# DensNet-121
class DensBlock(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(DensBlock, self).__init__()
        out_channel_1 = 4*growth_rate
        out_channel_2 = growth_rate

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel_1)
        self.conv1 = nn.Conv2d(in_channel, out_channel_1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channel_1, out_channel_2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        dense_connection = x
        x = F.relu(self.bn1(x)) # pre-activation
        x = self.conv1(x) # first layer in a block
        x = F.relu(self.bn2(x))
        x = self.conv2(x) # second layer in a block
        x = torch.cat((x, dense_connection), dim=1) # channel-wise concat
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channel):
        super(TransitionBlock, self).__init__()
        out_channel = in_channel // 2

        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(stride=2, kernel_size=2, padding=0)

    def forward(self, x):
        x = F.relu(self.bn(x)) # pre-activation
        x = self.conv1(x)
        x = self.pool(x)
        return x

class DensNet_100_12(nn.Module):
    def __init__(self, input_channels):
        super(DensNet_100_12, self).__init__()

        # initial conv, pool total 1
        self.bn = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, out_channels=24, kernel_size=3, stride=1, padding=1) # used 24 3x3 filter for 32x32 input.
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ## initial pooling not used for 32x32 input.

        # dense layers total 98: 96(dense) + 2(trans)
        self.dense_layers = self.repeat_block(in_channel=24, growth_rate=12, bundle_structure= [16, 16, 16])

        # final fully-connected layer total 1
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(342, 10) # 3-24-/216-108/-/300-150/-/342/

    def repeat_block(self, in_channel, growth_rate, bundle_structure):
        '''
        in_channel: number of channels of the input that enters the denseblock. => 2*growthrate (paper)
        bundle_structure: number of denseblocks for each bundle.
        '''
        bundles = list(range(len(bundle_structure))) # making list of index of bundle.
        total_dense_layers = [] # extracting all layers from each bundles and list them in one list.

        for bundle, i in zip(bundles, bundle_structure):
            for _ in range(i):
                total_dense_layers.append(DensBlock(in_channel, growth_rate))
                in_channel += growth_rate
            if bundle != bundles[-1]: # doesn't append transition block for the last denseblock
                total_dense_layers.append(TransitionBlock(in_channel))
                in_channel //= 2
        return nn.Sequential(*total_dense_layers)

    def forward(self, x):
        # initial conv, pooling layer
        x = F.relu(self.bn(x))
        x = self.conv1(x)
        # x = self.pool(x) ## not used for 32x32 input

        # dense layers
        x = self.dense_layers(x)

        # flatten, classification layer
        x = self.GAP(x) # shape: (batch-size, 342, 8, 8) -> (batch-size, 342, 1, 1)
        x = torch.flatten(x, 1) # shape: (batch-size, 342, 1, 1) -> (batch-size, 342)
        x = self.fc(x)
        return x

def load_model(name, input_channels, image_size):
    if name == "LeNet5":
        return LeNet5(input_channels, image_size)
    elif name == "ResNet18":
        return ResNet18(input_channels)
    elif name == "DensNet_100_12":
        return DensNet_100_12(input_channels)
    else:
        raise ValueError("Invalid model name")