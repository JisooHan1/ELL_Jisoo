import torch
import torch.nn as nn
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LogitNormLoss
class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04):
        super(LogitNormLoss, self).__init__()
        self.tau = tau

    def forward(self, logits, labels):
        # get output(logits) of the model
        logits = logits.to(device)  # (batch x channel)
        labels = labels.to(device)  # (batch)

        # normalize the output
        magnitude = F.normalize(logits, p=2, dim=1)  # (batch x 1)
        logit_norm = logits / (magnitude + 1e-7)  # (batch x channel)

        # logit_norm_loss
        logit_norm_loss = F.cross_entropy(logit_norm / self.tau, labels)
        return logit_norm_loss

# LogitNormLoss training config
logitnorm_config = {
    "criterion": LogitNormLoss,
    "lr": 0.1,
    "epochs": 200,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "optimizer": torch.optim.SGD,
    "scheduler_type": "step",
    "scheduler": torch.optim.lr_scheduler.MultiStepLR,
    "milestones": [80, 140],
    "gamma": 0.1
}
