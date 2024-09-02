import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Join(nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, path_list):
        # make a list of outcomes of each path after drop-path
        outputs = []
        for path in path_list:
            outputs.append(path)

        # joining by elementwise means
        join_outcome = sum(outputs)/len(outputs)

        # print(f'Join output shape: {join_outcome.shape}')  # Debugging output
        return join_outcome

class FractalBlock1Col(nn.Module):
    def __init__(self, output_channel, shared_conv):
        super(FractalBlock1Col, self).__init__()

        # use shared_conv filter
        self.conv1 = shared_conv

        # conv-bn-relu
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        # print(f'FractalBlock1Col output shape: {x.shape}')  # Debugging output
        return [x]

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_col, shared_conv):
        super(FractalBlock, self).__init__()

        # make a list of 0/1 for path1, path2: generate path vs doesn't generate path
        keep_prob = 0.85
        self.drop_keep = []
        [self.drop_keep.append((torch.rand(1) + keep_prob).floor().item()) for _ in range(2)] # drop = 0, keep = 1
        # print(f'1111111111111self.drop_keep: {self.drop_keep}')  # Debugging output

        # doesn't generate path2 if num_col==1
        if num_col == 1:
            self.drop_keep = [1, 0]
            # print(f'2222222222222222self.drop_keep: {self.drop_keep}')  # Debugging output

        # if both branch is dropped, choose one randomly
        elif sum(self.drop_keep) == 0:
            self.drop_keep[random.randint(0,1)] = 1
            # print(f'333333333333333333333self.drop_keep: {self.drop_keep}')  # Debugging output

        # branch1
        if self.drop_keep[0] == 1:
            self.path1 = self.generate_path1(output_channel, shared_conv)
            # print('PATH1 GENERATED')  # Debugging output

        # branch2(Ommited if C=1): FractalBlock-Join-FractalBlock
        if self.drop_keep[1] == 1:
            self.path2 = self.generate_path2(input_channel, output_channel, num_col, shared_conv)
            # print('PATH2 GENERATED')  # Debugging output

    def generate_path1(self, output_channel, shared_conv):
        return nn.Sequential(FractalBlock1Col(output_channel, shared_conv))

    def generate_path2(self, input_channel, output_channel, num_col, shared_conv):
        shared_conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.path2 = nn.Sequential(
            FractalBlock(input_channel, output_channel, num_col-1, shared_conv),
            Join(),
            FractalBlock(output_channel, output_channel, num_col-1, shared_conv2)
        )
        return self.path2

    def forward(self, x):
        # make a list of outputs from each path(structure)
        output_paths = []

        # print(f'444444444444444444self.drop_keep: {self.drop_keep}')  # Debugging output

        # output from path1
        if self.drop_keep[0] == 1:
            out1 = self.path1(x)
            output_paths.extend(out1)
        # print(f'FractalBlock path1 output shape: {out1[0].shape}')  # Debugging output

        # output form path2
        if self.drop_keep[1] == 1:
            out2 = self.path2(x)
            output_paths.extend(out2)
            # print(f'FractalBlock path2 output shape: {out2[0].shape}')  # Debugging output

        return output_paths

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 non-overlapping max-pooling

    def forward(self, paths):
        pool_outcome = []
        for i in range(len(paths)):
            out = self.pool(paths[i])
            pool_outcome.append(out)
            # print(f'ParallelPool output shape [{i}]: {out.shape}')  # Debugging output
        return pool_outcome

class FractalNet(nn.Module):
    def __init__(self, input_channel, output_channel, num_col):
        super(FractalNet, self).__init__()

        layers = []
        shared_conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)

        for i in range(1, 6):
            # block-pool-join
            layers.append(FractalBlock(input_channel, output_channel, num_col, shared_conv))
            layers.append(Pool())
            layers.append(Join())

            # adjust num of channel for next block
            input_channel = output_channel
            if i < 4:
                output_channel *= 2
            shared_conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)

        # total layer
        self.total_layer = nn.Sequential(*layers)

        # fc layer
        self.fc = nn.Linear(512, 10) # 512 = "output_channel"

    def forward(self, x):
        x = self.total_layer(x)
        # print(f'Final layer output shape: {x.shape}')  # Debugging output
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x