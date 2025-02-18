import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet18

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CSIResNet18(nn.Module):
    def __init__(self, input_channels, num_classes, pre_activation=True, projection_dim=128):
        super(CSIResNet18, self).__init__()

        self.backbone = ResNet18(input_channels, num_classes, pre_activation)
        self.backbone.fc = nn.Identity()
        self.projection_head = ProjectionHead(512, projection_dim)
        self.classifier = nn.Linear(projection_dim, 4)  # 4 classes: 0, 90, 180, 270

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        embeddings = self.projection_head(features)  # (8N, 128)
        rotation_predictions = self.classifier(features)  # (8N, 4)

        if return_features:
            return embeddings, rotation_predictions, features
        return embeddings, rotation_predictions
