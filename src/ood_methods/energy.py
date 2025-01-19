from .base_ood import BaseOOD
import torch

class Energy(BaseOOD):
    def __init__(self, model):
        self.model = model

    def apply_method(self, id_train_loader):
        pass

    def ood_score(self, images):
        outputs = self.model(images)  # (batch x class)
        return torch.logsumexp(outputs, dim=1)
