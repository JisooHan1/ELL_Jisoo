import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DenseNet(nn.Module):
    def __init__(self, input_channels):
        super(DenseNet, self).__init__()

        # initial conv, pool total 1
        self.conv1 = nn.Conv2d(input_channels, out_channels=24, kernel_size=3, stride=1, padding=1) # used 24 3x3 filter for 32x32 input.
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ## initial pooling not used for 32x32 input.

        # dense layers total 98: 96(dense) + 2(trans)
        self.dense_layers = self.repeat_block(in_channel=24, growth_rate=12, bundle_structure= [16, 16, 16])
        self.bn = nn.BatchNorm2d(342)

        # final fully-connected layer total 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
        x = self.conv1(x)  # x = self.pool(x) ## not used for 32x32 input
        x = self.dense_layers(x)
        x = F.relu(self.bn(x))  # (batch_size, 342, 8, 8)
        x = self.avgpool(x) # (batch_size, 342, 1, 1)
        x = torch.flatten(x, 1) # (batch_size, 342)
        x = self.fc(x) # (batch_size, 10)
        return x