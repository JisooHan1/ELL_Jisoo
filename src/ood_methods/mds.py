from .base_ood import BaseOOD
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS(BaseOOD):
    def __init__(self, model):
        super().__init__(model)
        self.num_classes = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_features = {cls: [] for cls in range(self.num_classes)}
        self.id_cls_means = []
        self.id_cls_covariances = None

    # # hook
    # def hook_function(self):
    #     def hook(_model, _input, output):
    #         self.penultimate_layer = self.avg_pool(output).squeeze()  # (batch x channel)
    #     return hook

    # method
    def get_class_features(self, id_dataloader):
        for inputs, labels in id_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.model(inputs)
            output = self.penultimate_layer.flatten(1)  # (batch x channel)

            for i, label in enumerate(labels):
                class_index = label.item()
                self.class_features[class_index].append(output[i])  # output[i] : (channel)

        # self.class_features is a dictionary, so we can't directly print its shape
        # Instead, let's print the shape of each class's features
        for cls in range(self.num_classes):
            print(f"Class {cls} features shape: {len(self.class_features[cls])} samples")

        return self.class_features

    def get_cls_means(self, class_features):
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample x channel)
            self.id_cls_means.append(torch.mean(class_data, dim=0))  # (channel)

        self.id_cls_means = torch.stack(self.id_cls_means)  # Convert list to tensor
        print(f"Class means tensor shape: {self.id_cls_means.shape}")  # (num_classes x channel)

        return self.id_cls_means

    def get_cls_covariances(self, class_features):
        class_stacks = []  # [(sample x channel), ...]
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample x channel)
            class_stacks.append(class_data)

        total_stack = torch.cat(class_stacks, dim=0)  # (total_sample, channel)
        N = total_stack.shape[0]

        class_deviations = torch.tensor([], device=device)
        for cls in range(self.num_classes):
            deviations = class_stacks[cls] - self.id_cls_means[cls].unsqueeze(0)  # (sample x channel)
            class_deviations = torch.cat([class_deviations, deviations], dim=0)  # (total_sample x channel)

        self.id_cls_covariances = torch.einsum('ni,nj->ij', class_deviations, class_deviations) / N  # (channel x channel)
        return self.id_cls_covariances

    # apply method
    def apply_method(self, id_loader):
        self.get_class_features(id_loader)
        self.get_cls_means(self.class_features)
        self.get_cls_covariances(self.class_features)

    # compute ood score
    def ood_score(self, inputs):
        inputs = inputs.to(device)
        self.model(inputs)

        output = self.penultimate_layer.flatten(1)  # (batch x channel)
        id_cls_means = self.id_cls_means  # (class x channel)
        id_cls_covariances = self.id_cls_covariances  # (channel x channel)
        print("output shape:", output.shape)
        print("id_cls_means shape:", id_cls_means.shape)
        print("id_cls_covariances shape:", id_cls_covariances.shape)

        det = torch.det(id_cls_covariances)
        if abs(det) < 1e-6:
            print("determinant too small")


        batch_deviations = output.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x channel)
        inv_covariance = torch.linalg.inv(id_cls_covariances)  # (channel x channel)
        print("batch_deviations shape:", batch_deviations.shape)
        print("inv_covariance shape:", inv_covariance.shape)
        mahalanobis_distances = torch.einsum('bci,ij,bcj->bc', batch_deviations,
                                             inv_covariance, batch_deviations)  # (batch x class)

        confidence_scores = torch.max(-mahalanobis_distances, dim=1)[0]
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
