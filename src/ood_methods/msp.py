from .base_ood import BaseOOD
import torch
import torch.nn.functional as F

class MSP(BaseOOD):
    def __init__(self, model):
        self.model = model

    def apply_method(self, id_train_loader):
        pass

    def ood_score(self, images):
        outputs = self.model(images)  # (batch x class)
        softmax_scores = F.softmax(outputs, dim=1)  # (batch x class)
        msp, _ = torch.max(softmax_scores, dim=1)  # (batch)
        return msp
