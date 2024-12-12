import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KNN:
    def __init__(self, model, k=50):
        self.model = model
        self.penultimate_layer = {}
        self.features = torch.tensor([], device=device)
        self.l2_features = torch.tensor([], device=device)
        self.k = k
        self.register_hooks()

    def get_activation(self, layer_name):
        def hook(_model, _input, output):
            self.penultimate_layer[layer_name] = output
        return hook

    def register_hooks(self):
        self.model.GAP.register_forward_hook(self.get_activation('penultimate'))

    def get_features(self, dataloader):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            self.model(inputs)

            self.features = torch.cat([self.features, self.penultimate_layer['penultimate'].flatten(1)])  # (batch, 512)
        self.l2_features = F.normalize(self.features, p=2, dim=1)  # (batch, 512)
        return self.l2_features  # (batch, 512)

    def knn_score(self, inputs, model=None):
        inputs = inputs.to(device)
        self.model(inputs)

        features = self.penultimate_layer['penultimate'].flatten(1)  # (batch, 512)
        l2_features = F.normalize(features, p=2, dim=1)

        distances = torch.cdist(self.l2_features, l2_features)  # (batch, batch)
        distances, _ = torch.sort(distances, dim=1, descending=False)

        kth_distance = distances[self.k -1]

        return -kth_distance  # (batch,)