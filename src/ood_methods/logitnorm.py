import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogitNorm:
    def __init__(self, model):
        self.model = model
        self.penultimate_layer = {}
        self.samples = torch.tensor([], device=device)
        self.register_hooks()

    def get_activation(self, layer_name):
        def hook(_model, _input, output):
            self.penultimate_layer[layer_name] = output
        return hook

    def register_hooks(self):
        self.model.GAP.register_forward_hook(self.get_activation('penultimate'))

    def get_samples(self, dataloader):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            self.model(inputs)
            self.samples = torch.cat([self.samples, self.penultimate_layer['penultimate'].flatten(1)])
            self.l2_samples = F.normalize(self.samples, p=2, dim=1)
        return self.l2_samples  # (num_samples, 512)

    def logitnorm_score(self, inputs, model=None):
        inputs = inputs.to(device)
        self.model(inputs)

        penultimate = self.penultimate_layer['penultimate'].flatten(1)  # (batch, 512)
        clamped = torch.clamp(penultimate, max=self.c)  # (batch, 512)
        logits = self.model.fc(clamped)  # (batch, 10)

        softmax = F.softmax(logits, dim=1)  # (batch, 10)
        scores = torch.max(softmax, dim=1)[0]  # (batch,)

        return scores  # (num_samples,)