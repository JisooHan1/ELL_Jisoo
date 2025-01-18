import torch
import torch.nn as nn
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OutlierExposureLoss
class OutlierExposureLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(OutlierExposureLoss, self).__init__()
        self.alpha = alpha  # 0.5

    def forward(self, id_outputs, id_labels, oe_outputs):
        # get output(logits) of the model
        id_outputs, id_labels, oe_outputs = id_outputs.to(device), id_labels.to(device), oe_outputs.to(device)  # (batch x class)

        # ID loss
        ID_loss = F.cross_entropy(id_outputs, id_labels)

        # OE loss: cross entropy with uniform distribution
        oe_softmax = F.softmax(oe_outputs, dim=1)  # (batch x class)
        uniform_dist = torch.ones_like(oe_softmax) / oe_softmax.size(1)  # (batch x class)
        OE_loss = (-torch.log(oe_softmax) * uniform_dist).mean()  # a value

        # total loss
        total_loss = ID_loss + self.alpha * OE_loss
        return total_loss

# Outlier Exposure training config
oe_config = {
    "criterion": OutlierExposureLoss,
    "lr": 0.01,
    "epochs": 10,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "optimizer": torch.optim.SGD,
    "scheduler": torch.optim.lr_scheduler.MultiStepLR,
    "milestones": [5, 8],
    "gamma": 0.1
}
