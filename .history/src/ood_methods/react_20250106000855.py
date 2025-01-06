from .base_ood import BaseOOD
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReAct(BaseOOD):
    def __init__(self, model, quantile=0.9):
        super().__init__(model)
        self.id_activations = torch.tensor([], device=device)
        self.c = None
        self.quantile = quantile

    # method

    # total channel
    # def get_activations(self, id_loader):
    #     for inputs, _ in id_loader:
    #         inputs = inputs.to(device)
    #         self.model(inputs)
    #         self.id_activations = torch.cat([self.id_activations, self.penultimate_layer.flatten(1)])  # (num_samples x channel)
    #     self.id_activations = self.id_activations.flatten()  # (num_samples * channel) = (total_channel)
    #     return self.id_activations

    # def calculate_c(self, id_activations):
    #     activations_np = id_activations.cpu().numpy()  # (total_channel)
    #     c_theshold = np.quantile(activations_np, self.quantile)  # (1)
    #     self.c = torch.tensor(c_theshold, device=device)


    # each channel
    def get_activations(self, id_loader):
        for inputs, _ in id_loader:
            inputs = inputs.to(device)
            self.model(inputs)
            self.id_activations = torch.cat([self.id_activations, self.penultimate_layer.flatten(1)])
        return self.id_activations  # (num_samples x channel)

    def calculate_c(self, id_activations):
        activations_np = id_activations.cpu().numpy()  # (num_samples x channel)
        c_theshold = np.quantile(activations_np, self.quantile, axis=0)  # (channel)
        self.c = torch.tensor(c_theshold, device=device)  # (channel)

        # Activation 통계
        print(f"\nActivation statistics:")
        print(f"Min activation: {id_activations.min().item():.4f}")
        print(f"Max activation: {id_activations.max().item():.4f}")
        print(f"Mean activation: {id_activations.mean().item():.4f}")
        print(f"Std activation: {id_activations.std().item():.4f}")

        # Threshold 통계
        print(f"\nThreshold (c) statistics:")
        print(f"Min threshold: {self.c.min().item():.4f}")
        print(f"Max threshold: {self.c.max().item():.4f}")
        print(f"Mean threshold: {self.c.mean().item():.4f}")
        print(f"Std threshold: {self.c.std().item():.4f}")

    # apply method: preprocessing before computing ood score
    def apply_method(self, id_loader):
        self.get_activations(id_loader)
        self.calculate_c(self.id_activations)

    # compute ood score
    def ood_score(self, inputs):
        self.model(inputs)

        activations = self.penultimate_layer.flatten(1)  # (batch x channel)
        clamped = torch.clamp(activations, max=self.c.to(activations.dtype))  # (batch x channel)
        logits = self.model.fc(clamped)  # (batch x num_classes)

        softmax = F.softmax(logits, dim=1)  # (batch x num_classes)
        scores = torch.max(softmax, dim=1)[0]  # (batch)

        return scores  # (batch)