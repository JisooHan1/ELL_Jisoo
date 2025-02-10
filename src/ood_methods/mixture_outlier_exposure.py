import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for CutMix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# OutlierExposureLoss with MixUp and CutMix
class MixtureOutlierExposureLoss(nn.Module):
    def __init__(self, lamda=0.5, mix_op='cutmix', alpha=1.0):
        super(MixtureOutlierExposureLoss, self).__init__()
        self.lamda = lamda  # Weight for OE loss
        self.mix_op = mix_op  # mixup or cutmix
        self.alpha = alpha  # Parameter for Beta distribution

    def forward(self, id_outputs, id_labels, oe_outputs, id_inputs, oe_inputs):
        id_outputs, id_labels, oe_outputs = id_outputs.to(device), id_labels.to(device), oe_outputs.to(device)
        id_inputs, oe_inputs = id_inputs.to(device), oe_inputs.to(device)

        # Apply MixUp or CutMix
        lam = np.random.beta(self.alpha, self.alpha)
        if self.mix_op == 'cutmix':
            bbx1, bby1, bbx2, bby2 = rand_bbox(id_inputs.size(), lam)
            id_inputs[:, :, bbx1:bbx2, bby1:bby2] = oe_inputs[:, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (id_inputs.size(-1) * id_inputs.size(-2)))
        elif self.mix_op == 'mixup':
            id_inputs = lam * id_inputs + (1 - lam) * oe_inputs

        # ID loss
        id_loss = F.cross_entropy(id_outputs, id_labels)

        # OE loss: cross entropy with uniform distribution
        oe_loss = -(oe_outputs.mean(1) - torch.logsumexp(oe_outputs, dim=1)).mean()

        # Total loss
        total_loss = id_loss + self.lamda * oe_loss
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
    "mix_op": "cutmix",  # Choose between "cutmix" and "mixup"
    "alpha": 1.0,  # Beta distribution parameter for mixup/cutmix
}
