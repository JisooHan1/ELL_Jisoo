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
            self.id_activations = torch.cat([self.id_activations, self.penultimate_layer.flatten(1)])  # (num_samples x channel)
        self.id_activations = self.id_activations.flatten()  # (num_samples * channel) = (total_channel)
        return self.id_activations

    def calculate_c(self, id_activations):
        activations_np = id_activations.cpu().numpy()  # (total_channel)
        c_theshold = np.quantile(activations_np, self.quantile)  # (1)
        self.c = torch.tensor(c_theshold, device=device)


    # def get_activations(self, id_loader):
    #     for inputs, _ in id_loader:
    #         inputs = inputs.to(device)
    #         self.model(inputs)
    #         self.id_activations = torch.cat([self.id_activations, self.penultimate_layer.flatten(1)])
    #     return self.id_activations  # (num_samples x 512)

    # def calculate_c(self, id_activations):
    #     activations_np = id_activations.cpu().numpy()  # (num_samples x channel)
    #     c_theshold = np.quantile(activations_np, self.quantile, axis=0)  # (channel)
    #     self.c = torch.tensor(c_theshold, device=device)  # (channel)

        # # 채널별 통계 출력
        # print("\nChannel-wise statistics:")
        # print(f"Min values per channel: [{activations_np.min(axis=0).min():.4f}, {activations_np.min(axis=0).max():.4f}]")
        # print(f"Max values per channel: [{activations_np.max(axis=0).min():.4f}, {activations_np.max(axis=0).max():.4f}]")
        # print(f"Mean values per channel: [{activations_np.mean(axis=0).min():.4f}, {activations_np.mean(axis=0).max():.4f}]")
        # print(f"Std values per channel: [{activations_np.std(axis=0).min():.4f}, {activations_np.std(axis=0).max():.4f}]")

        # Threshold 통계
        print(f"\nThreshold (c) statistics:")
        print(f"Min threshold: {self.c.min().item():.4f}")
        print(f"Max threshold: {self.c.max().item():.4f}")
        print(f"Mean threshold: {self.c.mean().item():.4f}")
        print(f"Std threshold: {self.c.std().item():.4f}")

    # apply method
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