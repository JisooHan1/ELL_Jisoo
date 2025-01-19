import torch
import torch.nn as nn
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OutlierExposureLoss
class OutlierExposureLoss(nn.Module):
    def __init__(self, lamda=0.5):
        super(OutlierExposureLoss, self).__init__()
        self.lamda = lamda  # Weight for OE loss

    def forward(self, id_outputs, id_labels, oe_outputs):
        # get output(logits) of the model
        id_outputs, id_labels, oe_outputs = id_outputs.to(device), id_labels.to(device), oe_outputs.to(device)

        # ID loss
        id_loss = F.cross_entropy(id_outputs, id_labels)

        # OE loss: cross entropy with uniform distribution
        oe_mean = oe_outputs.mean(1)  # Mean of logits across classes
        oe_logsumexp = torch.logsumexp(oe_outputs, dim=1)  # Logsumexp for logits
        oe_loss = 0.5 * -(oe_mean - oe_logsumexp).mean()  # Loss calculation as per the paper

        # total loss
        total_loss = id_loss + self.lamda * oe_loss
        return total_loss

# Outlier Exposure training config
oe_config = {
    "criterion": OutlierExposureLoss,
    "lr": 0.1,
    "epochs": 10,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "optimizer": torch.optim.SGD,
    "scheduler_type": "cosine",
    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
    "T_max": 10,
    "eta_min": 0,
    "milestones": [5, 8],
    "gamma": 0.1
}
