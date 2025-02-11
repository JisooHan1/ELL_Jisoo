import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SoftCrossEntropy
class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(soft_targets * log_probs, dim=-1)
        return loss.mean()

# OutlierExposureLoss with MixUp and CutMix
class MixtureOutlierExposureLoss(nn.Module):
    def __init__(self, lamda=0.5):
        super(MixtureOutlierExposureLoss, self).__init__()
        self.lamda = lamda  # Weight for OE loss
        self.soft_ce = SoftCrossEntropy()

    def forward(self, id_outputs, id_labels, mixed_outputs, ratio):
        id_outputs, id_labels, mixed_outputs = id_outputs.to(device), id_labels.to(device), mixed_outputs.to(device)

        num_classes = mixed_outputs.shape[1]
        one_hot_labels = F.one_hot(id_labels, num_classes=num_classes).to(device)
        uniform_labels = torch.ones_like(id_labels) / num_classes
        soft_targets = ratio * one_hot_labels + (1 - ratio) * uniform_labels

        # ID loss
        id_loss = F.cross_entropy(id_outputs, id_labels)

        # MOE loss: cross entropy with soft targets
        moe_loss = self.soft_ce(mixed_outputs, soft_targets)

        # Total loss
        total_loss = id_loss + self.lamda * moe_loss
        return total_loss


# Outlier Exposure training config
moe_config = {
    "criterion": MixtureOutlierExposureLoss,
    "lr": 0.01,
    "epochs": 10,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "optimizer": torch.optim.SGD,
    "scheduler_type": "cosine",
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "T_max": 10,
    "eta_min": 0,
    "milestones": [5, 8],
    "gamma": 0.1,
    "mix_op": "mixup"  # Choose between "cutmix" and "mixup"
}
