from .base_ood import BaseOOD
import torch
import torch.nn.functional as F

class ODIN(BaseOOD):
    def __init__(self, model, temperature=1000, epsilon=0.001):
        self.model = model
        self.temperature = temperature
        self.epsilon = epsilon

    def ood_score(self, inputs):
        inputs = inputs.clone().detach().requires_grad_(True)


        outputs = self.model(inputs) / self.temperature # (batch_size, num_classes)

        softmax_scores = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(softmax_scores, dim=1)

        loss = F.cross_entropy(outputs, pred_class)
        loss.backward()

        perturbed_input = inputs - self.epsilon * torch.sign(inputs.grad)

        with torch.no_grad():
            perturbed_output = self.model(perturbed_input) / self.temperature
        calibrated_scores = F.softmax(perturbed_output, dim=1) # (batch_size, num_classes)
        odin_score = torch.max(calibrated_scores, dim=1)[0] # (batch_size)

        return odin_score