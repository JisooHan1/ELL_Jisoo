from .base_ood import BaseOOD
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS(BaseOOD):
    def __init__(self, model):
        super().__init__(model)
        self.num_classes = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_features = {cls: [] for cls in range(self.num_classes)}
        self.id_cls_means = []
        self.id_cls_covariances = None

    # hook
    def hook_function(self):
        def hook(_model, _input, output):
            self.penultimate_layer = self.avg_pool(output).squeeze()  # (batch x channel)
        return hook

    # method
    def get_class_features(self, id_dataloader):
        for inputs, labels in id_dataloader:
            inputs, labels = inputs, labels
            self.model(inputs)
            output = self.penultimate_layer  # (batch x channel)

            for i, label in enumerate(labels):
                class_index = label.item()
                self.class_features[class_index].append(output[i])  # output[i] : (channel)
        return self.class_features

    def get_cls_means(self, class_features):
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample x channel)
            self.id_cls_means.append(torch.mean(class_data, dim=0))  # (channel)
        return self.id_cls_means

    def get_cls_covariances(self, class_features):
        class_stacks = []  # [(sample x channel), ...]
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample x channel)
            class_stacks.append(class_data)

        total_stack = torch.cat(class_stacks, dim=0)  # (total_sample, channel)
        N = total_stack.shape[0]

        class_covariances = []
        for cls in range(self.num_classes):
            deviations = class_stacks[cls] - self.id_cls_means[cls].unsqueeze(0)  # (sample x channel)
            class_covariances.append(torch.einsum('ni,nj->ij', deviations, deviations))

        self.id_cls_covariances = torch.stack(class_covariances).sum(dim=0) / N
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
        output = self.penultimate_layer.cpu()  # (batch x channel)
        cls_means = torch.stack(self.id_cls_means).cpu()  # (class x channel)
        cls_covariances = self.id_cls_covariances.cpu()  # (channel x channel)

        batch_deviations = output.unsqueeze(1) - cls_means.unsqueeze(0)  # (batch, class, channel)
        inv_covariance = torch.inverse(cls_covariances)  # (channel, channel)
        mahalanobis_distances = torch.einsum('bij,jk,bik->bi', batch_deviations,
                                             inv_covariance, batch_deviations)  # (batch, class)

        confidence_scores = torch.max(-mahalanobis_distances, dim=1)[0]
        return confidence_scores

