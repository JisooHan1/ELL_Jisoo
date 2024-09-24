import random
import torch
import torch.nn as nn

'''
로컬, 글로벌 드롭을 사용하지 않아도 약간의 성능 개선만 있을 뿐 정체 현상은 유지됨...
보다 근본적인 구조적 문제가 있음!!
'''

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

    def forward(self, paths):
        if self.num_col == 1:
            return paths[0]

        # Join - elementwise means
        stacked_paths = torch.stack(paths, dim=0)  # (num_paths, batch, channel, height, width)
        out = torch.mean(stacked_paths, dim=0)  # (batch, channel, height, width)
        return out


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate):
        super(BasicBlock, self).__init__()
        self.conv_bn_drop = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
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
        self.path1 = BasicBlock(input_channel, output_channel, dropout_rate)
        if num_col > 1:
            self.path2 = nn.ModuleList([
                FractalBlock(input_channel, output_channel, num_col - 1, dropout_rate),
                Join(num_col - 1),
                FractalBlock(output_channel, output_channel, num_col - 1, dropout_rate),
            ])
        else:
            self.path2 = None

    def forward(self, x):
        # List of outputs from each path
        output_paths = []
        output_paths.extend(self.path1(x))  # Add path 1
        if self.path2 is not None:
            for layer in self.path2:
                x = layer(x)
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
            self.layers.append(FractalBlock(input_channel, output_channel, self.num_col, dropout_rates[i]))
            if i != 3:  # Add Pool layer except for the last block
                self.layers.append(Pool())
            self.layers.append(Join(self.num_col))

            # Adjust channels for the next block
            input_channel = output_channel
            output_channel *= 2

        # Final fc layer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)  # 512 = "output_channel"

    def forward(self, x):
        # Choose sampling in "batch level": local vs global
        for layer in self.layers:
            x = layer(x)
        x = self.GAP(x)  # (batch-size, 512, 8, 8) -> (batch-size, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch-size, 512, 1, 1) -> (batch-size, 512)
        x = self.fc(x)
        return x
