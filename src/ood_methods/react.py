from .base_ood import BaseOOD
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReAct(BaseOOD):
    def __init__(self, model, percentile=90):
        super().__init__(model)
        self.id_train_activations = None  # (total_id_train_samples x channel)
        self.c = None  # tensor of a value
        self.percentile = percentile

    # method
    def get_activations(self, id_train_loader):
        id_train_activations = []
        for images, _ in id_train_loader:
            images = images.to(device)
            self.model(images)
            id_train_activations.append(self.penultimate_layer)
        self.id_train_activations = torch.cat(id_train_activations).flatten()  # (total_id_train_samples * channel)

    # total channel quantile
    def calculate_c(self, id_activations):
        activations_np = id_activations.cpu().numpy()  # (total_id_train_samples * channel)
        c_theshold = np.percentile(activations_np, self.percentile)  # tensor of a value
        self.c = torch.tensor(c_theshold, device=device).float()

        # Activation statistics
        print(f"\nActivation statistics:")
        print(f"c: {self.c.item():.4f}")
        print(f"Min activation: {id_activations.min().item():.4f}")
        print(f"Max activation: {id_activations.max().item():.4f}")
        print(f"Mean activation: {id_activations.mean().item():.4f}")
        print(f"Std activation: {id_activations.std().item():.4f}")

    # apply method: preprocessing before computing ood score
    def apply_method(self, id_train_loader):
        self.get_activations(id_train_loader)
        self.calculate_c(self.id_train_activations)

    # compute ood score
    def ood_score(self, images):
        self.model(images)
        rectified_activations = torch.minimum(self.penultimate_layer, self.c)  # (batch x channel)
        rectified_logits = self.model.fc(rectified_activations)  # (batch x class)
        return torch.logsumexp(rectified_logits, dim=1)  # energy score
        # softmax = F.softmax(rectified_logits, dim=1)  # (batch x class)
        # scores, _ = torch.max(softmax, dim=1)  # (batch)
        # return scores  # (batch)