# resnet-18 pre-activation
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
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

class ResNet(nn.Module):
    def __init__(self, input_channels):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1) # used 3x3 kernel for 32x32 input
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2) ## initial pooling ommited for 32x32 input
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bundle1 = self.repeat_block(64, 64, 1)
        self.bundle2 = self.repeat_block(64, 128, 2)
        self.bundle3 = self.repeat_block(128, 256, 2)
        self.bundle4 = self.repeat_block(256, 512, 2)
        self.bn = nn.BatchNorm2d(512)

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
        x = F.relu(self.bn(x))  # batch normalization and relu added before fc
        x = self.avgpool(x)  # (batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch x 512)
        x = self.fc(x) # layer 18  # (batch x 10)
        return x

# # resnet-18 post-activation
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):
#         super(ResBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channel)  # Moved after conv1
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel)  # Moved after conv2

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channel != out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channel))  # BatchNorm applied after Conv in shortcut

#     def forward(self, x):
#         identity = self.shortcut(x)
#         x = self.conv1(x)  # Convolution first
#         x = self.bn1(x)   # BatchNorm after Conv
#         x = F.relu(x)     # ReLU last

#         x = self.conv2(x) # Second Convolution
#         x = self.bn2(x)   # BatchNorm after Conv
#         x += identity     # Add shortcut
#         x = F.relu(x)     # ReLU applied last
#         return x

# class ResNet(nn.Module):
#     def __init__(self, input_channels):
#         super(ResNet, self).__init__()

#         self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1)  # First Conv layer
#         self.bn1 = nn.BatchNorm2d(64)  # BatchNorm for initial Conv
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         self.bundle1 = self.repeat_block(64, 64, 1)
#         self.bundle2 = self.repeat_block(64, 128, 2)
#         self.bundle3 = self.repeat_block(128, 256, 2)
#         self.bundle4 = self.repeat_block(256, 512, 2)
#         self.bn2 = nn.BatchNorm2d(512)  # BatchNorm for final layer

#         self.fc = nn.Linear(512, 10)

#     def repeat_block(self, in_channel, out_channel, stride):
#         bundle = []
#         bundle.append(ResBlock(in_channel, out_channel, stride))
#         bundle.append(ResBlock(out_channel, out_channel, 1))
#         return nn.Sequential(*bundle)

#     def forward(self, x):
#         x = self.conv1(x)        # Initial Convolution
#         x = self.bn1(x)          # BatchNorm after Conv
#         x = F.relu(x)            # ReLU last

#         x = self.bundle1(x)      # Layer 2~5
#         x = self.bundle2(x)      # Layer 6~9
#         x = self.bundle3(x)      # Layer 10~13
#         x = self.bundle4(x)      # Layer 14~17
#         x = self.bn2(x)          # BatchNorm for penultimate layer
#         x = F.relu(x)            # ReLU after BatchNorm

#         x = self.avgpool(x)      # Adaptive average pooling
#         x = torch.flatten(x, 1)  # Flatten (batch x 512)
#         x = self.fc(x)           # Fully connected layer
#         return x

# # resnet-34 post-activation
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):
#         super(ResBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channel)  # BatchNorm after conv1
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel)  # BatchNorm after conv2

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channel != out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channel))  # Shortcut connection

#     def forward(self, x):
#         identity = self.shortcut(x)
#         x = self.conv1(x)  # Convolution first
#         x = self.bn1(x)   # BatchNorm after Conv
#         x = F.relu(x)     # ReLU last

#         x = self.conv2(x)  # Second Convolution
#         x = self.bn2(x)    # BatchNorm after Conv
#         x += identity      # Add shortcut
#         x = F.relu(x)      # ReLU applied last
#         return x

# class ResNet(nn.Module):
#     def __init__(self, input_channels, num_classes=10):
#         super(ResNet, self).__init__()

#         self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1)  # First Conv layer
#         self.bn1 = nn.BatchNorm2d(64)  # BatchNorm for initial Conv
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         # Define ResNet-34 layer configuration
#         self.bundle1 = self.repeat_block(64, 64, 3, 1)
#         self.bundle2 = self.repeat_block(64, 128, 4, 2)
#         self.bundle3 = self.repeat_block(128, 256, 6, 2)
#         self.bundle4 = self.repeat_block(256, 512, 3, 2)
#         self.bn2 = nn.BatchNorm2d(512)  # BatchNorm for final layer

#         self.fc = nn.Linear(512, num_classes)

#     def repeat_block(self, in_channel, out_channel, num_blocks, stride):
#         """Create repeated blocks for each layer."""
#         bundle = []
#         bundle.append(ResBlock(in_channel, out_channel, stride))  # First block with stride
#         for _ in range(1, num_blocks):
#             bundle.append(ResBlock(out_channel, out_channel, 1))  # Subsequent blocks with stride=1
#         return nn.Sequential(*bundle)

#     def forward(self, x):
#         x = self.conv1(x)        # Initial Convolution
#         x = self.bn1(x)          # BatchNorm after Conv
#         x = F.relu(x)            # ReLU last

#         x = self.bundle1(x)      # Layer 2~5
#         x = self.bundle2(x)      # Layer 6~9
#         x = self.bundle3(x)      # Layer 10~15
#         x = self.bundle4(x)      # Layer 16~18
#         x = self.bn2(x)          # BatchNorm for penultimate layer
#         x = F.relu(x)            # ReLU after BatchNorm

#         x = self.avgpool(x)      # Adaptive average pooling
#         x = torch.flatten(x, 1)  # Flatten (batch x 512)
#         x = self.fc(x)           # Fully connected layer
#         return x
