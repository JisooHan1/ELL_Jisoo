import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(ResNet, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Building residual blocks
        self.bundle1 = self._make_layer(64, 64, 2, stride=1)
        self.bundle2 = self._make_layer(64, 128, 2, stride=2)
        self.bundle3 = self._make_layer(128, 256, 2, stride=2)
        self.bundle4 = self._make_layer(256, 512, 2, stride=2)

        # Adaptive average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.bundle1(x)
        x = self.bundle2(x)
        x = self.bundle3(x)
        x = self.bundle4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):
#         super(ResBlock, self).__init__()

#         self.bn1 = nn.BatchNorm2d(in_channel)
#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channel != out_channel:
#             self.shortcut = nn.Sequential(
#                 nn.BatchNorm2d(in_channel),
#                 nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False))

#     def forward(self, x):
#         identity = self.shortcut(x)
#         x = self.conv1(F.relu(self.bn1(x))) # first layer (pre-activation)
#         x = self.conv2(F.relu(self.bn2(x))) # second layer
#         x += identity
#         return x

# class ResNet(nn.Module):
#     def __init__(self, input_channels):
#         super(ResNet, self).__init__()

#         self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1) # used 3x3 kernel for 32x32 input
#         # self.pool = nn.MaxPool2d(kernel_size=3, stride=2) ## initial pooling ommited for 32x32 input
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         self.bundle1 = self.repeat_block(64, 64, 1)
#         self.bundle2 = self.repeat_block(64, 128, 2)
#         self.bundle3 = self.repeat_block(128, 256, 2)
#         self.bundle4 = self.repeat_block(256, 512, 2)
#         self.bn = nn.BatchNorm2d(512)

#         self.fc = nn.Linear(512, 10)

#     def repeat_block(self, in_channel, out_channel, stride):
#         bundle = []
#         bundle.append(ResBlock(in_channel, out_channel, stride))
#         bundle.append(ResBlock(out_channel, out_channel, 1))
#         return nn.Sequential(*bundle)

#     def forward(self, x):
#         x = self.conv1(x) # layer 1
#         # x = self.pool(x) ## initial pooling ommited for 32x32 input
#         x = self.bundle1(x) # layer 2~5
#         x = self.bundle2(x) # layer 6~9
#         x = self.bundle3(x) # layer 10~13
#         x = self.bundle4(x) # layer 14~17
#         x = F.relu(self.bn(x))  # batch normalization and relu added before fc
#         x = self.avgpool(x)  # (batch, 512, 1, 1)
#         x = torch.flatten(x, 1)  # (batch x 512)
#         x = self.fc(x) # layer 18  # (batch x 10)
#         return x