import torch
import torch.nn as nn
import random

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 non-overlapping max-pooling

    def forward(self, paths, *args):
        pool_outcome = []
        device = paths[0].device

        for i in range(len(paths)):
            out = self.pool(paths[i].to(device))
            pool_outcome.append(out.to(device))
        return pool_outcome

# List -> List
def local_drop(paths, drop_prob):
    device = paths[0].device
    # Drop path
    drop_mask = torch.rand(len(paths), device=device) > drop_prob # True: keep, False: drop
    results = [path for path, keep in zip(paths, drop_mask) if keep.item()]

    # Kept path
    if len(results) == 0: # Handle all dropped
        results.append(random.choice(paths))
    return results

def global_drop(paths, chosen_col): # chosen_col: 4/3/2/1
    device = paths[0].device

    if len(paths) < chosen_col:
        return [torch.zeros_like(paths[0], device=device)]
    else:
        kept_path_index = len(paths) - chosen_col
        return [paths[kept_path_index].to(device)]

class Join(nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, paths, sampling):
        device = paths[0].device

        if sampling == 'local': # Local Sampling
            paths = local_drop(paths, drop_prob=0.15)
        elif isinstance(sampling, tuple): # Global Sampling
            chosen_col = sampling[1]
            paths = global_drop(paths, chosen_col)
        paths = [p.to(device) for p in paths]

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
        device = x.device
        x = x.to(device)
        return [self.conv_bn_drop(x)]

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
        self.path1 = gen_path1(input_channel, output_channel, dropout_rate)
        self.path2 = gen_path2(input_channel, output_channel, num_col, dropout_rate) if num_col > 1 else None

    def forward(self, x, sampling):
        device = x.device
        # List of outputs from each path
        output_paths = []
        output_paths.extend(self.path1(x)) # Add path 1
        if self.path2 is not None:
            for layer in self.path2:
                x = layer(x, sampling)
            output_paths.extend(x) # Add path 2
        return output_paths

class FractalNet(nn.Module):
    def __init__(self, input_channel):
        super(FractalNet, self).__init__()

        output_channel=64
        self.num_col=4
        dropout_rates = [0, 0.1, 0.2, 0.3, 0.4] if self.training else [0, 0, 0, 0, 0]

        # 4 blocks: block-pool-join x4
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(FractalBlock(input_channel, output_channel, self.num_col, dropout_rates[i]))
            if i != 3:  # Add Pool layer except for the last block
                self.layers.append(Pool())
            self.layers.append(Join())

            # Adjust channels for the next block
            input_channel = output_channel
            output_channel *= 2

        # Final fc layer
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10) # 512 = "output_channel"

    def forward(self, x):
        # Choose sampling in "batch level": local vs global
        sampling = 'local' if random.random() <= 0.5 else ('global', random.randint(1, self.num_col))
        for layer in self.layers:
            x = layer(x, sampling)
        x = self.GAP(x) # (batch-size, 512, 8, 8) -> (batch-size, 512, 1, 1)
        x = torch.flatten(x, 1) # (batch-size, 512, 1, 1) -> (batch-size, 512)
        x = self.fc(x)
        return x
