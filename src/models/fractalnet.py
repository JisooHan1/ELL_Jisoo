import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FractalBlock1Col(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate):
        super(FractalBlock1Col, self).__init__()

        # conv-bn-relu-dropout
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn(x))
        x = self.dropout(x)
        return [x]

class LocalSampling:
    def __init__(self, drop_prob, num_col):
        # make a list of 0/1 for path1, path2: generate path vs doesn't generate path ex)[0,1]
        keep_prob = 1-drop_prob
        self.drop_keep_list = []
        [self.drop_keep_list.append((torch.rand(1) + keep_prob).floor().item()) for _ in range(2)] # drop = 0, keep = 1

        # handling condition 1) doesn't generate path2 if "num_col==1"
        if num_col == 1:
            self.drop_keep_list = [1, 0]

        # handling condition 2) if "both branch is dropped", choose one randomly
        elif sum(self.drop_keep_list) == 0:
            self.drop_keep_list[random.randint(0,1)] = 1

    def sampling_result(self):
        return self.drop_keep_list

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_col, dropout_rate):
        super(FractalBlock, self).__init__()

        local_sampling = LocalSampling(drop_prob=0.15, num_col=num_col)
        self.drop_keep_list = local_sampling.sampling_result()

        # generate branch1
        if self.drop_keep_list[0] == 1:
            self.path1 = self.generate_path1(input_channel, output_channel, dropout_rate)
        # generate branch2(Ommited if C=1): FractalBlock-Join-FractalBlock
        if self.drop_keep_list[1] == 1:
            self.path2 = self.generate_path2(input_channel, output_channel, num_col, dropout_rate)

    def generate_path1(self, input_channel, output_channel, dropout_rate):
        return nn.Sequential(FractalBlock1Col(input_channel, output_channel, dropout_rate))

    def generate_path2(self, input_channel, output_channel, num_col, dropout_rate):
        self.path2 = nn.Sequential(
            FractalBlock(input_channel, output_channel, num_col-1, dropout_rate),
            Join(),
            FractalBlock(output_channel, output_channel, num_col-1, dropout_rate)
        )
        return self.path2

    def forward(self, x):
        # make a list of outputs from each path(structure)
        output_paths = []

        # output from path1
        if self.drop_keep_list[0] == 1:
            out1 = self.path1(x)
            output_paths.extend(out1)

        # output form path2
        if self.drop_keep_list[1] == 1:
            out2 = self.path2(x)
            output_paths.extend(out2)
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

class Join(nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, path_list):
        # make a list of outcomes after drop-path
        outputs = []
        for path in path_list:
            outputs.append(path)

        # join by elementwise means
        join_outcome = sum(outputs)/len(outputs)
        return join_outcome

class FractalNet(nn.Module):
    def __init__(self, input_channel):
        super(FractalNet, self).__init__()

        output_channel=64
        num_col=4
        layers = []
        dropout_rates = [0,0.1,0.2,0.3,0.4] if self.training == True else [0,0,0,0,0]

        # 5 blocks
        for i in range(1, 6):
            # block-pool-join
            dropout_rate = dropout_rates[i-1]
            layers.append(FractalBlock(input_channel, output_channel, num_col, dropout_rate))
            layers.append(Pool())
            layers.append(Join())

            # adjust num of channel for next block
            input_channel = output_channel
            if i < 4:
                output_channel *= 2

        # total fractal layer of 5 blocks
        self.total_layer = nn.Sequential(*layers)

        # final fc layer
        self.fc = nn.Linear(512, 10) # 512 = "output_channel"

    def forward(self, x):
        x = self.total_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x