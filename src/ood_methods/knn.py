from .base_ood import BaseOOD
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KNN(BaseOOD):
    def __init__(self, model, k=50):
        super().__init__(model)
        self.id_features = torch.tensor([], device=device)
        self.k = k

    # get normalized id features
    def get_features(self, id_loader):
        for inputs, _ in id_loader:
            self.model(inputs)
            self.id_features = torch.cat([self.id_features, self.penultimate_layer.flatten(1)])
        self.id_features = F.normalize(self.id_features, p=2, dim=1)  # (num_samples x channel)
        return self.id_features

    # apply method
    def apply_method(self, id_loader):
        self.get_features(id_loader)

    # compute ood score
    def ood_score(self, inputs):
        inputs = inputs.to(device)
        self.model(inputs)

        features = self.penultimate_layer.flatten(1)  # (batch x channel)
        l2_features = F.normalize(features, p=2, dim=1)

        distances = torch.cdist(l2_features, self.id_features)  # (batch x num_samples)
        distances, _ = torch.sort(distances, dim=1, descending=False)

        kth_distance = distances[self.k - 1]

        return -kth_distance  # (batch)