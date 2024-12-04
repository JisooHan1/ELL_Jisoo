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
        self.penultimate_outputs = {}  # {'penultimate': (batch, channel)}
        self.num_classes = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_features = {cls: [] for cls in range(self.num_classes)}
        self.cls_means = []
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
            self.get_activation('penultimate', self.penultimate_outputs)
        )

    def remove_hook(self):
        self.model.GAP.remove_forward_hook(
            self.get_activation('penultimate', self.penultimate_outputs)
        )

    def get_class_features(self, id_dataloader):
        for inputs, labels in id_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            self.model(inputs)
            output = self.penultimate_outputs['penultimate']  # (batch, channel)

            for i, label in enumerate(labels):
                class_index = label.item()
                self.class_features[class_index].append(output[i])  # output[i] : (channel,)

        return self.class_features

    def get_cls_means(self, class_features):
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample, channel)
            self.cls_means.append(torch.mean(class_data, dim=0))  # (channel)
        return self.cls_means

    def get_cls_covariances(self, class_features):
        class_stacks = []  # [(sample, channel), ...]
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample, channel)
            class_stacks.append(class_data)

        total_stack = torch.cat(class_stacks, dim=0)  # (total_sample, channel)
        N = total_stack.shape[0]
        total_mean = torch.mean(total_stack, dim=0)  # (channel,)

        # Calculate deviations from total mean
        deviations = total_stack - total_mean.unsqueeze(0)  # (total_sample, channel)

        # Fix the einsum operation for covariance calculation
        self.cls_covariances = torch.einsum('ni,nj->ij', deviations, deviations) / N  # (channel, channel)

        return self.cls_covariances

    def get_mds_scores(self, test_dataloader, cls_means, cls_covariances, epsilon=0.001):
        confidence_scores = torch.tensor([], device=device)
        cls_means = [mean.clone().detach().to(device) for mean in cls_means]
        cls_covariances = cls_covariances.clone().detach().to(device)

        inv_covariance = torch.inverse(cls_covariances)

        for inputs, _ in test_dataloader:
            inputs = inputs.to(device).clone().detach().requires_grad_(True)

            if inputs.grad is not None:
                inputs.grad.zero_()

            self.model(inputs)
            output = self.penultimate_outputs['penultimate']  # (batch, channel)

            batch_deviations = output.unsqueeze(1) - torch.stack(cls_means).unsqueeze(0)  # (batch, num_classes, channel)
            mahalanobis_distances = torch.einsum('bij,jk,bik->bi', batch_deviations, inv_covariance, batch_deviations)  # (batch, num_classes)
            c_hat = torch.argmin(mahalanobis_distances, dim=1)  # (batch,)

            # loss: batch average
            batch_size = mahalanobis_distances.shape[0]
            loss = mahalanobis_distances[torch.arange(batch_size), c_hat]  # (batch,)
            loss = loss.mean()
            loss.backward()

            # perturbed data
            perturbed_inputs = inputs - epsilon * torch.sign(inputs.grad)

            with torch.no_grad():
                self.model(perturbed_inputs)
                perturbed_output = self.penultimate_outputs['penultimate']  # (batch, channel)

                perturbed_batch_deviations = perturbed_output.unsqueeze(1) - torch.stack(cls_means).unsqueeze(0)  # (batch, num_classes, channel)
                perturbed_mahalanobis_distances = torch.einsum('bij,jk,bik->bi', perturbed_batch_deviations, inv_covariance, perturbed_batch_deviations)  # (batch, num_classes)
                score = torch.max(-perturbed_mahalanobis_distances, dim=1)[0]  # (batch,)

                confidence_scores = torch.cat([confidence_scores, score])

        return confidence_scores

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

    # load datasets
    id_trainset, id_testset = load_dataset("CIFAR10")
    _ood_trainset, ood_testset = load_dataset("SVHN")

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=0)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=0)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=0)

    # register hook
    detector.register_hook()

    # get class features, means, and covariances
    detector.get_class_features(id_train_loader)
    detector.get_cls_means(detector.class_features)
    detector.get_cls_covariances(detector.class_features)

    # get mds scores
    id_scores = detector.get_mds_scores(id_test_loader, detector.cls_means, detector.cls_covariances).to(device)
    ood_scores = detector.get_mds_scores(ood_test_loader, detector.cls_means, detector.cls_covariances).to(device)

    evaluate_ood_detection(id_scores, ood_scores)

    # remove hook
    detector.remove_hook()


if __name__ == '__main__':
    main()




