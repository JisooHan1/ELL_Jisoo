import torch
import torch.nn as nn
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSI loss
class CSILoss(nn.Module):
    def __init__(self, temperature=0.5, lamda=1.0):
        super(CSILoss, self).__init__()
        self.temperature = temperature
        self.lamda = lamda
        self.ce = nn.CrossEntropyLoss()

    def contrastive_loss(self, z):

        batch_size = z.shape[0]  # 8N

        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature  # (8N, 8N)

        pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool).to(device)
        for i in range(0, batch_size, 2):
            pos_mask[i, i+1] = 1
            pos_mask[i+1, i] = 1
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool).to(device)
        neg_mask = neg_mask & ~pos_mask

        positives = similarity_matrix[pos_mask].view(batch_size, 1)  # (8N, 1)
        negatives = similarity_matrix[neg_mask].view(batch_size, -1)  # (8N, 8N-2)

        logits = torch.cat([positives, negatives], dim=1)  # (8N, 8N-1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(device)  # (8N)

        loss = F.cross_entropy(logits / self.temperature, labels)
        return loss

    def rotation_loss(self, rotation_predictions, labels):
        return self.ce(rotation_predictions, labels % 4)

    def forward(self, outputs, labels):
        embeddings, rotation_predictions = outputs

        contrastive_loss = self.contrastive_loss(embeddings)
        rotation_loss = self.rotation_loss(rotation_predictions, labels)

        total_loss = contrastive_loss + self.lamda * rotation_loss
        return total_loss

# Outlier Exposure training config
csi_config = {
    "criterion": CSILoss,
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
