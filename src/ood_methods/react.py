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
    def get_activations(self, id_loader):
        for inputs, _ in id_loader:
            inputs = inputs.to(device)
            self.model(inputs)
            self.id_activations = torch.cat([self.id_activations, self.penultimate_layer.flatten(1)])
        return self.id_activations  # (num_samples x 512)

    def calculate_c(self, activations):
        activations_np = activations.cpu().numpy()
        c_theshold = np.quantile(activations_np, self.quantile)
        self.c = torch.tensor(c_theshold, device=device)

        # print statistics
        print(f"\nQuantile {self.quantile}:")
        print(f"Calculated threshold (c): {self.c.item():.4f}")
        print(f"Input samples range: [{activations_np.min():.4f}, {activations_np.max():.4f}]")
        print(f"Samples mean: {activations_np.mean():.4f}")
        print(f"Samples std: {activations_np.std():.4f}")

    # apply method
    def apply_method(self, id_loader):
        self.get_activations(id_loader)
        self.calculate_c(self.id_activations)

    # compute ood score
    def ood_score(self, inputs):
        self.model(inputs)

        activations = self.penultimate_layer.flatten(1)  # (batch x channel)
        clamped = torch.clamp(activations, max=self.c)  # (batch x channel)
        logits = self.model.fc(clamped)  # (batch x 10)

        softmax = F.softmax(logits, dim=1)  # (batch x 10)
        scores = torch.max(softmax, dim=1)[0]  # (batch)

        return scores  # (batch)