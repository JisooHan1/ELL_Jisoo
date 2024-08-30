import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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

class ResNet18(nn.Module):
    def __init__(self, input_channels):
        super(ResNet18, self).__init__()

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

# DensNet-100
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

class DensNet100(nn.Module):
    def __init__(self, input_channels):
        super(DensNet100, self).__init__()

        # initial conv, pool total 1
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
        x = self.conv1(x)
        # x = self.pool(x) ## not used for 32x32 input

        # dense layers
        x = self.dense_layers(x)

        # flatten, classification layer
        x = self.GAP(x) # shape: (batch-size, 342, 8, 8) -> (batch-size, 342, 1, 1)
        x = torch.flatten(x, 1) # shape: (batch-size, 342, 1, 1) -> (batch-size, 342)
        x = self.fc(x)
        return x

# FractalNet
class DropPath(nn.Module):
    def __init__(self, drop_probability):
        super(DropPath, self).__init__()

        self.keep_prob = 1 - drop_probability

    def forward(self, x):
        # doesn't apply when testing
        if self.training == False:
            return x

        # 0: drop, 1: keep (under drop_prob)
        mask_element = (torch.rand(1)+self.keep_prob).floor().item()

        # masking tensor => broadcasting expected
        mask  = torch.full((x.size(0), 1, 1, 1), mask_element, dtype=x.dtype, device=x.device)
        x = x * mask
        return x

class Join(nn.Module):
    def __init__(self, num_paths, drop_probability):
        super(Join, self).__init__()

        # make list of drop-path modules for each path
        self.drop_path_module = nn.ModuleList()
        [self.drop_path_module.append(DropPath(drop_probability)) for _ in range(num_paths)]

    def forward(self, path_list):
        # make a list of outcomes of each path after drop-path
        outputs = []
        for i, path in enumerate(path_list):
            out = self.drop_path_module[i](path)
            outputs.append(out)

        # joining by elementwise means
        join_outcome = sum(outputs)/len(path_list)

        # Local sampling: keep at least one path when join
        if join_outcome.sum() == 0:
            random_index = random.randint(0, len(path_list)-1)
            join_outcome = path_list[random_index]

        return join_outcome

class FractalBlock1Col(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FractalBlock1Col, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

class FractalBlock2Col(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FractalBlock2Col, self).__init__()

        # path1: conv
        self.path1 = nn.Sequential(FractalBlock1Col(input_channel, output_channel))

        # path2: conv-conv
        self.path2 = nn.Sequential(
            FractalBlock1Col(input_channel, output_channel),
            FractalBlock1Col(output_channel, output_channel),
        )

    def forward(self, x):
        # compute two path in parllel
        out1 = self.path1(x)
        out2 = self.path2(x)

        # make a list of outputs from each path
        paths_to_list = []
        paths_to_list.extend([out1, out2])
        return paths_to_list

class FractalBlock3Col(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FractalBlock3Col, self).__init__()

        # path1: conv
        self.path1 = nn.Sequential(FractalBlock1Col(input_channel, output_channel))

        # path2&3: FractalBlock2Col-Join-FractalBlock2Col
        self.path2 = nn.Sequential(
            FractalBlock2Col(input_channel, output_channel),
            Join(num_paths=2, drop_probability=0.15),
            FractalBlock2Col(output_channel, output_channel)
        )

    def forward(self, x):
        # compute two structure in parllel
        out1 = self.path1(x)
        out2 = self.path2(x)

        # make a list of outputs from each path(structure)
        paths_to_list = []
        paths_to_list.append(out1)
        paths_to_list.extend(out2)
        return paths_to_list

class FractalBlock4Col(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FractalBlock4Col, self).__init__()

        # path1: conv
        self.path1 = nn.Sequential(FractalBlock1Col(input_channel, output_channel))

        # path2&3: FractalBlock3Col-Join-FractalBlock3Col
        self.path2 = nn.Sequential(
            FractalBlock3Col(input_channel, output_channel),
            Join(num_paths=3, drop_probability=0.15),
            FractalBlock3Col(output_channel, output_channel)
        )

    def forward(self, x):
        # compute two structure in parllel
        out1 = self.path1(x)
        out2 =self.path2(x)

        # make a list of outputs from each path(structure)
        paths_to_list = []
        paths_to_list.append(out1)
        paths_to_list.extend(out2)
        return paths_to_list

class ParallelPool(nn.Module):
    def __init__(self, num_cols):
        super(ParallelPool, self).__init__()

        self.num_cols = num_cols
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 non-overlapping max-pooling

    def forward(self, paths):
        pool_outcome = []
        for i in range(self.num_cols):
            out = self.pool(paths[i])
            pool_outcome.append(out)
        return pool_outcome

class FractalNet(nn.Module):
    def __init__(self, input_channel, output_channel, num_col):
        super(FractalNet, self).__init__()

        # Initialize block based on number of columns
        if num_col == 2:
            block = FractalBlock2Col
        elif num_col == 3:
            block = FractalBlock3Col
        elif num_col == 4:
            block = FractalBlock4Col
        else:
            raise ValueError("Invalid number of columns.")

        layers = []

        for i in range(5):
            # block-pool-join
            layers.append(block(input_channel, output_channel))
            layers.append(ParallelPool(num_cols=num_col))
            layers.append(Join(num_paths=num_col, drop_probability=0.15))

            # adjust num of channel for next block
            input_channel = output_channel
            if i != 3:
                output_channel *= 2

        # total layer
        self.total_layer = nn.Sequential(*layers)

        # fc layer
        self.fc = nn.Linear(512, 10) # 512 = "output_channel"

    def forward(self, x):
        x = self.total_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_model(name, input_channels, image_size):
    if name == "LeNet5":
        return LeNet5(input_channels, image_size)
    elif name == "ResNet18":
        return ResNet18(input_channels)
    elif name == "DensNet100":
        return DensNet100(input_channels)
    elif name == "FractalNet":
        return FractalNet(input_channels, output_channel=64, num_col=4)
    else:
        raise ValueError("Invalid model name")











#     class RecursiveBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, columns):
#         super(RecursiveBlock, self).__init__()

#         # base conv: single layer
#         self.conv = baseblock(in_channel, out_channel)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.is_base = 0
#         self.columns = columns

#         # check if it's a base case
#         if columns == 1:
#             self.is_base = 1

#         # path1: doubling the original structure
#         if columns >= 1:
#             doubling_original_path = []
#             [doubling_original_path.append(RecursiveBlock(in_channel, out_channel, columns-1)) for _ in range(2)]
#             doubling_original_path.append(self.pool)
#             self.path1 = nn.Sequential(*doubling_original_path)

#         # path2: adding parallel single layer
#             parallel_single_layer = []
#             parallel_single_layer.append(baseblock(in_channel, out_channel), self.pool)
#             self.path2 = nn.Sequential(*parallel_single_layer)

#     def forward(self, x):
#         # only use conv in the first base case.
#         if self.is_base == 1:
#             x = self.conv(x)
#         # join two
#         if len(self.path1) != 1:
#             out1 = self.path1(x)
#             out2 = self.path2(x)
#             x = (out1*(self.columns - 1) + out2) / self.columns
#         return x

