import random
import torch
import torch.nn as nn


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


class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 non-overlapping max-pooling

    def forward(self, paths):
        pooled = []
        [pooled.append(self.pool(path)) for path in paths]
        return pooled


class Join(nn.Module):
    def __init__(self, num_col):
        super(Join, self).__init__()

        self.num_col = num_col

    def forward(self, paths, sampling):
        if self.num_col == 1:
            return paths[0]
        elif self.training:
            if sampling == "local":  # Local Sampling
                paths = local_drop(paths, drop_prob=0.1)
            elif isinstance(sampling, tuple):  # Global Sampling
                chosen_col = sampling[1]
                paths = global_drop(paths, chosen_col)

        # Join - elementwise means
        stacked_paths = torch.stack(paths, dim=0)  # (num_paths, batch, channel, height, width)
        out = torch.mean(stacked_paths, dim=0)  # (batch, channel, height, width)
        return out


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate):
        super(BasicBlock, self).__init__()
        self.basic_layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return [self.basic_layer(x)]


class FractalBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_col, dropout_rate):
        super(FractalBlock, self).__init__()
        self.num_col = num_col
        self.path1 = BasicBlock(input_channel, output_channel, dropout_rate)
        if num_col > 1:
            self.path2 = nn.ModuleList([
                FractalBlock(input_channel, output_channel, num_col - 1, dropout_rate),
                Join(num_col - 1),
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
        dropout_rates = [0, 0.1, 0.15, 0.2] if self.training else [0, 0, 0, 0]

        # 3 blocks: block-pool-join x3
        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(FractalBlock(input_channel, output_channel, self.num_col, dropout_rates[i]))
            # Adjust channels for the next block
            input_channel = output_channel
            if i < 2:  # Add Pool layer except for the last block
                output_channel *= 2
                self.layers.append(Pool())
            self.layers.append(Join(self.num_col))

        # Final fc layer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)  # 256 = "output_channel"

    def forward(self, x):
        # Choose sampling in "batch level": local vs global
        block_count = 1  # To track block number
        sampling = ("local" if random.random() <= 0.5 else ("global", random.randint(1, self.num_col)))
        for layer in self.layers:
            x = layer(x, sampling)
        x = self.GAP(x)  # (batch-size, 256, 8, 8) -> (batch-size, 256, 1, 1)
        x = torch.flatten(x, 1)  # (batch-size, 256, 1, 1) -> (batch-size, 256)
        x = self.fc(x)
        return x
