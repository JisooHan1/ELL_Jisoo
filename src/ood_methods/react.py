import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReAct:
    def __init__(self, model, quantile=0.9):
        self.model = model
        self.penultimate_layer = {}
        self.samples = torch.tensor([], device=device)
        self.c = None
        self.quantile = quantile
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
        return self.samples  # (num_samples, 512)

    def calculate_threshold(self, samples):
        samples_np = samples.flatten().cpu().numpy()
        c_theshold = np.quantile(samples_np, self.quantile)
        self.c = torch.tensor(c_theshold, device=device)

        # 통계 출력
        print(f"\nQuantile {self.quantile}:")
        print(f"Calculated threshold (c): {self.c.item():.4f}")
        print(f"Input samples range: [{samples_np.min():.4f}, {samples_np.max():.4f}]")
        print(f"Samples mean: {samples_np.mean():.4f}")
        print(f"Samples std: {samples_np.std():.4f}")

    def react_score(self, inputs, model=None):
        inputs = inputs.to(device)
        self.model(inputs)

        penultimate = self.penultimate_layer['penultimate'].flatten(1)  # (batch, 512)
        clamped = torch.clamp(penultimate, max=self.c)  # (batch, 512)
        logits = self.model.fc(clamped)  # (batch, 10)

        softmax = F.softmax(logits, dim=1)  # (batch, 10)
        scores = torch.max(softmax, dim=1)[0]  # (batch,)

        return scores  # (batch,)