from .base_ood import BaseOOD
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReAct(BaseOOD):
    def __init__(self, model, quantile=0.95):
        super().__init__(model)
        self.c = None
        self.quantile = quantile

    # method
    def get_activations(self, id_train_loader):
        id_train_activations = []
        for images, _ in id_train_loader:
            images = images.to(device)
            self.model(images)
            id_train_activations.append(self.penultimate_layer)
        self.id_train_activations = torch.cat(id_train_activations)  # (total_id_train_samples x channel)
        return self.id_train_activations.view(-1)  # (total_id_train_samples * channel)

    # # each channel
    # def calculate_c(self, id_activations):
    #     activations_np = id_activations.cpu().numpy()  # (num_samples x channel)
    #     c_theshold = np.quantile(activations_np, self.quantile, axis=0)  # (channel)
    #     self.c = torch.tensor(c_theshold, device=device)  # (channel)

    # total channel quantile
    def calculate_c(self, id_activations):
        activations_np = id_activations.cpu().numpy()  # (total_id_train_samples * channel)
        c_theshold = np.quantile(activations_np, self.quantile)  # a value
        self.c = torch.tensor(c_theshold, device=device)

        # Activation 통계
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

        activations = self.penultimate_layer  # (batch x channel)
        clamped = torch.clamp(activations, max=self.c.to(activations.dtype))  # (batch x channel)
        logits = self.model.fc(clamped)  # (batch x num_classes)

        softmax = F.softmax(logits, dim=1)  # (batch x num_classes)
        scores = torch.max(softmax, dim=1)[0]  # (batch)

        return scores  # (batch)