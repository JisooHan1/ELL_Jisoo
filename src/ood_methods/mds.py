from .base_ood import BaseOOD
import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS(BaseOOD):
    def __init__(self, model):
        super().__init__(model)
        self.num_classes = 10
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.all_features = []
        self.all_labels = []
        self.id_cls_means = []
        self.id_covariances = []

    # # hook
    # def hook_function(self):
    #     def hook(_model, _input, output):
    #         self.penultimate_layer = self.avg_pool(output).squeeze()  # (batch x channel)
    #     return hook

    # method
    def get_class_features(self, id_dataloader):
        self.model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in id_dataloader:
                images, labels = images.to(device), labels.to(device)  # (batch)

                _ = self.model(images)
                output = self.penultimate_layer.flatten(1)  # (batch x channel x 1 x 1) -> (batch x channel)

                all_features.append(output.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                self.all_features = np.concatenate(all_features)
                self.all_labels = np.concatenate(all_labels)

        return self.all_features, self.all_labels

    def compute_gaussian_params(self, all_features, all_labels):
        means = []
        covariances = []
        for cls in range(self.num_classes):
            cls_features = all_features[all_labels == cls]
            means.append(np.mean(cls_features, axis=0))
            covariances.append(np.cov(cls_features, rowvar=False))
        self.id_cls_means = np.array(means)
        self.id_covariances = np.array(covariances)
        return self.id_cls_means, self.id_covariances

    # apply method
    def apply_method(self, id_loader):
        self.get_class_features(id_loader)
        self.compute_gaussian_params(self.all_features, self.all_labels)

    # compute ood score
    def ood_score(self, images):
        images = images.to(device)
        self.model(images)
        output = self.penultimate_layer.flatten(1)  # (batch x channel)

        id_cls_means = self.id_cls_means  # (class x channel)
        id_covariances = self.id_covariances  # (channel x channel)

        # det = torch.det(id_cls_covariances)
        # if abs(det) < 1e-6:
        #     print("determinant too small")


        batch_deviations = output.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x channel)
        inv_covariance = torch.linalg.inv(id_covariances)  # (channel x channel)
        mahalanobis_distances = torch.einsum('bci,ij,bcj->bc', batch_deviations,
                                             inv_covariance, batch_deviations)  # (batch x class)

        confidence_scores = torch.max(-mahalanobis_distances, dim=1)[0]  # (batch)
        return confidence_scores



# def ood_score(self, inputs, magnitude=0.001):
#     self.model(inputs)
#     output = self.penultimate_layer  # (batch x channel)
#     cls_means = torch.stack(self.id_cls_means).cpu()  # (class x channel)
#     cls_covariances = self.id_cls_covariances.cpu()  # (channel x channel)

#     # Mahalanobis distance calculation
#     batch_deviations = output.unsqueeze(1) - cls_means.unsqueeze(0)  # (batch, class, channel)
#     inv_covariance = torch.inverse(cls_covariances)  # (channel, channel)
#     mahalanobis_distances = torch.einsum('bij,jk,bik->bi', batch_deviations, inv_covariance, batch_deviations)

#     # Get the class with minimal distance
#     min_distance, _ = torch.min(mahalanobis_distances, dim=1)

#     # Compute gradient w.r.t inputs
#     inputs.requires_grad_()
#     loss = torch.mean(min_distance)  # Mahalanobis loss
#     loss.backward()

#     # Add noise based on the gradient
#     gradient = torch.sign(inputs.grad.data)  # Get the sign of the gradient
#     perturbed_inputs = inputs - magnitude * gradient  # Add perturbation

#     # Recompute Mahalanobis distance for perturbed inputs
#     self.model(perturbed_inputs)
#     perturbed_output = self.penultimate_layer
#     perturbed_deviations = perturbed_output.unsqueeze(1) - cls_means.unsqueeze(0)
#     perturbed_distances = torch.einsum('bij,jk,bik->bi', perturbed_deviations, inv_covariance, perturbed_deviations)

#     confidence_scores = -torch.min(perturbed_distances, dim=1)[0]
#     return confidence_scores
