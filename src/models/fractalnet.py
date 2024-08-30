import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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