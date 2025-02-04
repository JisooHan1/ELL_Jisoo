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
        oe_loss = torch.mean(-oe_outputs.mean(1) + torch.logsumexp(oe_outputs, dim=1))

        # total loss
        total_loss = id_loss + self.lamda * oe_loss
        return total_loss

# Outlier Exposure training config

# paper: wrs
oe_config = {
    "criterion": OutlierExposureLoss,
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
    "gamma": 0.1
}

# # resnet18
# oe_config = {
#     "criterion": OutlierExposureLoss,
#     "lr": 0.1,
#     "epochs": 30,  # 학습을 더 길게 (10 → 30)
#     "weight_decay": 1e-4,  # ResNet18에는 더 낮은 weight decay 사용
#     "momentum": 0.9,
#     "optimizer": torch.optim.SGD,

#     # MultiStepLR 적용 (Cosine Annealing 대신)
#     "scheduler_type": "step",
#     "scheduler": torch.optim.lr_scheduler.MultiStepLR,
#     "milestones": [15, 25],  # LR decay 시점 변경
#     "gamma": 0.2  # decay rate 증가 (0.1 → 0.2)
# }
