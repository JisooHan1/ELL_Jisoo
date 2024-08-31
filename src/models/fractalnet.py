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
        is_dropped = True if mask_element == 0 else False

        # masking tensor => broadcasting expected
        mask  = torch.full((x.size(0), 1, 1, 1), mask_element, dtype=x.dtype, device=x.device)
        x = x * mask
        return x, is_dropped

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
            out = self.drop_path_module[i](path)[0]
            if self.drop_path_module[i](path)[1] == False: # path not dropped
                outputs.append(out)

        outputs_sum = sum(outputs)
        # Local sampling: keep at least one path when join
        if outputs_sum.sum().item() == 0:
            random_index = random.randint(0, len(path_list)-1)
            join_outcome = path_list[random_index]

        # joining by elementwise means
        else:
            join_outcome = outputs_sum/len(outputs)

        return join_outcome

class FractalBlock1Col(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FractalBlock1Col, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return [x]

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_col):
        super(FractalBlock, self).__init__()

        for i in range(1, num_col+1):
            drop_prob = 0 if i == 2 else 0.15
            self.is_C_1 = False

            # branch1
            self.path1 = nn.Sequential(FractalBlock1Col(input_channel, output_channel))

            # branch2(Ommited if C=1): FractalBlock-Join-FractalBlock
            if i == 1:
                self.is_C_1 = True
            else:
                self.path2 = nn.Sequential(
                    FractalBlock(input_channel, output_channel, i-1),
                    Join(num_paths=i-1, drop_probability=drop_prob),
                    FractalBlock(output_channel, output_channel, i-1)
                )
    def forward(self, x):
        # make a list of outputs from each path(structure)
        output_paths = []

        # output from path1
        out1 = self.path1(x)
        output_paths.extend(out1)

        # output form path2
        if self.is_C_1 == False:
            out2 = self.path2(x)
            output_paths.extend(out2)
        return output_paths

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

        layers = []

        for i in range(1, 5+1):
            # block-pool-join
            layers.append(FractalBlock(input_channel, output_channel, num_col))
            layers.append(ParallelPool(num_cols=num_col))
            layers.append(Join(num_paths=num_col, drop_probability=0.15))

            # adjust num of channel for next block
            input_channel = output_channel
            if i != 4:
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