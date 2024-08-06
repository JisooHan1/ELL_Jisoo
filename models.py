import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, input_channels, image_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5) #  1or3 input cannel, 6 output channel, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        out_size = ((image_size-4)//2 - 4)//2
        self.fc1 = nn.Linear(16 * out_size * out_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(name, input_channels, image_size):
    if name == "LeNet5":
        return LeNet5(input_channels, image_size)
    else:
        raise ValueError("Invalid model name")
