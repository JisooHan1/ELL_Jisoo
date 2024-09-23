import random

import torch
import torch.nn as nn


class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 2x2 non-overlapping max-pooling

    def forward(self, paths, *args):
        pool_outcome = []
        for i in range(len(paths)):
            x = self.pool(paths[i])
            pool_outcome.append(x)
        return pool_outcome


def local_drop(paths, drop_prob):
    results = [path for path in paths if torch.rand(1).item() > drop_prob]  # drop-path

    if not results:  # Handle all dropped
        results.append(random.choice(paths))
    return results


def global_drop(paths, chosen_col):  # chosen_col: 4/3/2/1

    if len(paths) < chosen_col:
        return [torch.zeros_like(paths[0])]
    else:
        kept_path_index = len(paths) - chosen_col
        return [paths[kept_path_index]]


class Join(nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, paths, sampling):
        if sampling == "local":  # Local Sampling
            paths = local_drop(paths, drop_prob=0.15)
        elif isinstance(sampling, tuple):  # Global Sampling
            chosen_col = sampling[1]
            paths = global_drop(paths, chosen_col)

        # Join - elementwise means
        stacked_paths = torch.stack(
            paths, dim=0
        )  # (num_paths, batch, channel, height, width)
        out = torch.mean(stacked_paths, dim=0)  # (batch, channel, height, width)
        return out


# conv-bn-relu-dropout
class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate):
        super(BasicBlock, self).__init__()
        self.conv_bn_drop = nn.Sequential(
            nn.Conv2d(
                input_channel, output_channel, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return [self.conv_bn_drop(x)]


class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_col, dropout_rate):
        super(FractalBlock, self).__init__()
        self.num_col = num_col
        self.path1 = nn.Sequential(
            BasicBlock(input_channel, output_channel, dropout_rate)
        )
        if num_col > 1:
            self.path2 = nn.ModuleList([
                FractalBlock(input_channel, output_channel, num_col - 1, dropout_rate),
                Join(),
                FractalBlock(output_channel, output_channel, num_col - 1, dropout_rate),
            ])
        else:
            self.path2 = None

    def forward(self, x, sampling):
        # List of outputs from each path
        output_paths = []
        output_paths.extend(self.path1(x))  # Add path 1
        if self.path2 is not None:
            for layer in self.path2:
                x = layer(x, sampling)
            output_paths.extend(x)  # Add path 2
        return output_paths


class FractalNet(nn.Module):
    def __init__(self, input_channel):
        super(FractalNet, self).__init__()

        output_channel = 64
        self.num_col = 4
        dropout_rates = [0, 0.1, 0.2, 0.3, 0.4] if self.training else [0, 0, 0, 0, 0]

        # 4 blocks: block-pool-join x4
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(
                FractalBlock(
                    input_channel, output_channel, self.num_col, dropout_rates[i]
                )
            )
            if i != 3:  # Add Pool layer except for the last block
                self.layers.append(Pool())
            self.layers.append(Join())

            # Adjust channels for the next block
            input_channel = output_channel
            output_channel *= 2

        # Final fc layer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)  # 512 = "output_channel"

    def forward(self, x):
        # Choose sampling in "batch level": local vs global
        sampling = (
            "local"
            if random.random() <= 0.5
            else ("global", random.randint(1, self.num_col))
        )
        for layer in self.layers:
            x = layer(x, sampling)
        x = self.GAP(x)  # (batch-size, 512, 8, 8) -> (batch-size, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch-size, 512, 1, 1) -> (batch-size, 512)
        x = self.fc(x)
        return x
