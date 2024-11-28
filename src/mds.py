import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from models import ResNet, DenseNet
import argparse
from torcheval.metrics import BinaryAUROC, BinaryAUPRC
from ood_methods import get_ood_methods

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.id_layer_outputs = {}  # {'penultimate': (batch, channel)}
        self.ood_layer_outputs = {}  # {'penultimate': (batch, channel)}
        self.num_classes = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_features = {cls: [] for cls in range(self.num_classes)}
        self.cls_means = {cls: None for cls in range(self.num_classes)}
        self.cls_covariances = None

    def load_model(self, model_path):
        model = DenseNet(3).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def get_activation(self, layer_name, output_dict):
        def hook(_model, _input, output):
            pooled_output = self.avg_pool(output).squeeze().to(device)
            output_dict[layer_name] = pooled_output.detach().to(device)
        return hook

    def register_hook(self):
        self.model.GAP.register_forward_hook(
            self.get_activation('penultimate', self.id_layer_outputs)
        )

    def remove_hook(self):
        self.model.GAP.remove_forward_hook(
            self.get_activation('penultimate', self.id_layer_outputs)
        )

    def get_class_features(self, id_dataloader):
        for inputs, labels in id_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.model(inputs)
            output = self.id_layer_outputs['penultimate']  # (batch, channel)
            for i, label in enumerate(labels):
                class_index = label.item()
                self.class_features[class_index].append(output[i].to(device))
        return self.class_features

    def get_cls_means(self, class_features):
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0).to(device)  # (n_samples, channel)
            self.cls_means[cls] = torch.mean(class_data, dim=0)  # (channel)
        return self.cls_means

    def get_cls_covariances(self, class_features, cls_means):
        class_stacks = []  # [(n_samples, channel), ...]
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0).to(device)  # (n_samples, channel)
            cls_means[cls] = torch.mean(class_data, dim=0)  # (channel)
            class_stacks.append(class_data)

        total_stack = torch.cat(class_stacks, dim=0).to(device)  # (total_samples, channel)
        total_mean = torch.mean(total_stack, dim=0).to(device)  # (channel)
        class_stacks = torch.stack(class_stacks, dim=0).to(device)  # (num_classes, n_samples, channel)

        N = class_stacks.shape[0]  # Total number of samples
        deviations = class_stacks - total_mean.unsqueeze(0)  # (total_samples, channel)
        self.cls_covariances = torch.matmul(deviations.T, deviations) / N  # (channel, channel)
        return self.cls_covariances

    def get_mds_scores(self, ood_dataloader, cls_means, cls_covariances, epsilon=0.001):
        scores = []
        inv_covariance = torch.inverse(cls_covariances)  # (channel, channel)

        for inputs, _ in ood_dataloader:
            inputs = inputs.to(device)
            inputs.requires_grad = True

            self.model(inputs)
            output = self.ood_layer_outputs['penultimate']  # (batch, channel)

            cls_mean_tensor = torch.tensor([cls_means[i] for i in range(self.num_classes)], device=device)  # (num_classes, channel)
            batch_error = output.unsqueeze(1) - cls_mean_tensor.unsqueeze(0)  # (batch, num_classes, channel)

            mahalanobis_distances = torch.einsum('bij,jk,bik->bi', batch_error, inv_covariance, batch_error)  # (batch, num_classes)
            c_hat = torch.argmin(mahalanobis_distances, dim=1)  # (batch,)

            batch_size = mahalanobis_distances.size(0)
            loss = sum(mahalanobis_distances[i, c_hat[i]] for i in range(batch_size))/batch_size

            # perturbed data
            perturbed_inputs = inputs - epsilon * torch.sign(inputs.grad)
            inputs.grad = None
            self.model(perturbed_inputs)
            perturbed_output = self.ood_layer_outputs['penultimate']  # (batch, channel)

            perturbed_batch_error = perturbed_output.unsqueeze(1) - cls_mean_tensor.unsqueeze(0)  # (batch, num_classes, channel)
            perturbed_mahalanobis_distances = torch.einsum('bij,jk,bik->bi', perturbed_batch_error, inv_covariance, perturbed_batch_error)  # (batch, num_classes)
            perturbed_c_hat = torch.argmin(perturbed_mahalanobis_distances, dim=1)  # (batch,)

            scores.extend(perturbed_c_hat.tolist())

        return scores

def evaluate_ood_detection(id_scores, ood_scores):
    # generate list of label: ID = 1, OOD = 0
    labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)])
    scores = torch.cat([id_scores, ood_scores])

    # Use Binary metrics for OOD detection
    binary_auroc = BinaryAUROC()
    binary_auroc.update(scores, labels)
    binary_auprc = BinaryAUPRC()
    binary_auprc.update(scores, labels)

    auroc = binary_auroc.compute()
    aupr = binary_auprc.compute()

    print(f'AUROC: {auroc:.4f}')
    print(f'AUPR: {aupr:.4f}')


def main():
    model_path = "logs/DenseNet/trained_model/trained_resnet_20241031_004039.pth"
    detector = MDS(model_path)

    # register hook
    detector.register_hook()

    # load datasets
    id_trainset, id_testset = load_dataset("CIFAR10")
    _ood_trainset, ood_testset = load_dataset("SVHN")

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=0)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=0)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=0)

    detector.get_class_features(id_train_loader)
    detector.get_cls_means(detector.class_features)
    detector.get_cls_covariances(detector.class_features, detector.cls_means)

    id_scores = torch.tensor(detector.get_mds_scores(id_test_loader, detector.cls_means, detector.cls_covariances))
    ood_scores = torch.tensor(detector.get_mds_scores(ood_test_loader, detector.cls_means, detector.cls_covariances))

    evaluate_ood_detection(id_scores, ood_scores)

if __name__ == '__main__':
    main()




