import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 non-overlapping max-pooling

    def forward(self, paths, *args):
        pool_outcome = []
        for i in range(len(paths)):
            out = self.pool(paths[i])
            pool_outcome.append(out)
        return pool_outcome


# list -> list (dropped)
def local_drop(paths, drop_prob):
    results = paths[:] # for output
    # Drop path
    for i in range(len(paths)):
            if random.random() <= drop_prob:
                results[i] = 'dropped'
    # Kept path
    results = list(filter(lambda x: x != 'dropped', results))
    if len(results) == 0: # Handle all dropped
        results.append(random.choice(paths))
    return results


def global_drop(paths, chosen_col): # chosen_col: 4/3/2/1

    if len(paths) < chosen_col:
        results = [torch.zeros(paths[0].shape)]
        return results
    else:
        kept_path_index = len(paths) - chosen_col
        results = [paths[kept_path_index]]
        return results

class Join(nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, paths, sampling):
        # Local Sampling
        if sampling == 'local':
            paths = local_drop(paths, drop_prob=0.15)
        # Global Sampling
        if type(sampling) == tuple:
            chosen_col = sampling[1]
            paths = global_drop(paths, chosen_col)

        # Join - elementwise means
        stacked_paths = torch.stack(paths, dim=0)  # (num_paths, batch, channel, height, width)
        out = torch.mean(stacked_paths, dim=0)     # (batch, channel, height, width)
        return out

class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate):
        super(BasicBlock, self).__init__()
        # conv-bn-relu-dropout
        self.conv_bn_drop = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        device = next(self.parameters()).device
        return [self.conv_bn_drop(x.to(device))]

def gen_path1(input_channel, output_channel, dropout_rate):
    return nn.Sequential(BasicBlock(input_channel, output_channel, dropout_rate))

def gen_path2(input_channel, output_channel, num_col, dropout_rate):
    return nn.ModuleList([
        FractalBlock(input_channel, output_channel, num_col-1, dropout_rate),
        Join(),
        FractalBlock(output_channel, output_channel, num_col-1, dropout_rate)
    ])

class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_col, dropout_rate):
        super(FractalBlock, self).__init__()

        self.num_col = num_col

        # generate path1
        self.path1 = gen_path1(input_channel, output_channel, dropout_rate)

        # generate path2: (Ommited if C=1)
        if num_col > 1:
            self.path2 = gen_path2(input_channel, output_channel, num_col, dropout_rate)
        else:
            self.path2 = None

    def forward(self, x, sampling):

        # make a list of outputs from each path(branch)
        output_paths = []

        # output from path1
        output_paths.extend(self.path1(x))

        # output form path2
        if self.path2 is not None:
            for layer in self.path2:
                x = layer(x, sampling)
            output_paths.extend(x)

        return output_paths

class FractalNet(nn.Module):
    def __init__(self, input_channel):
        super(FractalNet, self).__init__()

        output_channel=64
        num_col=4

        dropout_rates = [0, 0.1, 0.2, 0.3, 0.4] if self.training else [0, 0, 0, 0, 0]

        # 4 blocks
        self.layers = nn.ModuleList()
        for i in range(1, 5):
            # block-pool-join
            dropout_rate = dropout_rates[i-1]
            self.layers.append(FractalBlock(input_channel, output_channel, num_col, dropout_rate))
            if i != 4:
                self.layers.append(Pool())
            self.layers.append(Join())

            # adjust num of channel for next block
            input_channel = output_channel
            output_channel *= 2

        # final fc layer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10) # 512 = "output_channel"

    def forward(self, x):
        # choose sampling in "batch level": local vs global
        sampling = 'local' if random.random() <= 0.5 else ('global', random.randint(1,4))

        for layer in self.layers:
            x = layer(x, sampling)
        x = self.GAP(x) # shape: (batch-size, 512, 8, 8) -> (batch-size, 512, 1, 1)
        x = torch.flatten(x, 1) # shape: (batch-size, 512, 1, 1) -> (batch-size, 512)
        x = self.fc(x)
        return x
