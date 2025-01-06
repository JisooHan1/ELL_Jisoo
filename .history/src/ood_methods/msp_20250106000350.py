from .base_ood import BaseOOD
import torch
import torch.nn.functional as F

class MSP(BaseOOD):
    def __init__(self, model):
        self.model = model

    def ood_score(self, inputs):
        outputs = self.model(inputs)  # (batch x num_class)
        softmax_scores = F.softmax(outputs, dim=1)  # (batch x num_class)
        msp, _ = torch.max(softmax_scores, dim=1)  # (batch)
        return msp
