from .base_ood import BaseOOD
import torch
import torch.nn.functional as F

class ODIN(BaseOOD):
    def __init__(self, model, temperature=1000, epsilon=0.0001):
        self.model = model
        self.temperature = temperature
        self.epsilon = epsilon

    def apply_method(self, id_train_loader):
        pass

    def ood_score(self, images):
        with torch.set_grad_enabled(True):
            images = images.clone().detach().requires_grad_(True)

            # temperature scaling applied output
            outputs = self.model(images) / self.temperature # (batch x num_class); channel == num_class

            softmax_scores = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(softmax_scores, dim=1)

            # adding preturbation to input images
            loss = F.cross_entropy(outputs, pred_class)
            loss.backward()
            perturbed_images = images - self.epsilon * torch.sign(images.grad)

        with torch.no_grad():
            perturbed_output = self.model(perturbed_images) / self.temperature
            calibrated_scores = F.softmax(perturbed_output, dim=1) # (batch x num_class)
            odin_score, _ = torch.max(calibrated_scores, dim=1) # (batch)

        return odin_score