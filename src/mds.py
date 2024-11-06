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

# Definition
id_layer_outputs = {}
ood_layer_outputs = {}
num_classes = 10
num_layers = 50  # 48 DenseBlock, 2 TransitionBlock
class_features = {layer: {cls: [] for cls in range(num_classes)} for layer in range(num_layers)}
cls_means = {layer: {cls: None for cls in range(num_classes)} for layer in range(num_layers)}
cls_covariances = {layer: None for layer in range(num_layers)}
avg_pool = nn.AdaptiveAvgPool2d((1, 1))

# Model loading function
def load_model(model_path):
    model = DenseNet(3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# Main script execution
if __name__ == '__main__':
    # Load model
    model = load_model("logs/DenseNet/trained_model/trained_resnet_20241031_004039.pth")

    # Load datasets
    id_trainset, id_testset = load_dataset("CIFAR10")
    ood_trainset, ood_testset = load_dataset("SVHN")

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=0)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=0)
    ood_train_loader = torch.utils.data.DataLoader(ood_trainset, batch_size=64, shuffle=True, num_workers=0)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=0)

    # Layer activation hook function
    def get_activation(layer_name, output_dict):
        def hook(model, input, output):
            pooled_output = avg_pool(output).squeeze(-1)  # (batch, channel, h, w) -> (batch, channel)
            output_dict[layer_name] = pooled_output
        return hook

    # Register hook for each layer
    for i in range(num_layers):
        model.dense_layers[i].register_forward_hook(get_activation(f"layer_{i}", id_layer_outputs))

    # Compute class means and covariances
    for inputs, labels in id_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model(inputs)
        for layer_index, (layer_name, output) in enumerate(id_layer_outputs.items()):
            for i, label in enumerate(labels):
                class_index = label.item()
                class_features[layer_index][class_index].append(output[i])

    for layer in range(num_layers):
        class_deviation_sums = []
        N = sum(len(class_features[layer][cls]) for cls in range(num_classes))

        for cls in range(num_classes):
            class_data = torch.cat(class_features[layer][cls], dim=0).to(device)
            cls_means[layer][cls] = torch.mean(class_data, dim=0)

            deviations = class_data - cls_means[layer][cls]
            sample_deviations = [torch.matmul(sample, sample.T) for sample in deviations]
            class_deviation_sums.append(sum(sample_deviations))

        cls_covariances[layer] = sum(class_deviation_sums) / N

    def mahalanobis_distance(x, mean, inversed_covariance):
        diff = x - mean
        return torch.matmul(diff, torch.matmul(inversed_covariance, diff.T))


    def mds_score(model, test_sample, epsilon, cls_means, cls_covariances, num_layers, alpha_weights):
        mds_score = []

        for i in range(num_layers):
            model.dense_layers[i].register_forward_hook(get_activation(f"layer_{i}", ood_layer_outputs))

        model(test_sample)

        test_sample = test_sample.to(device)

        for layer in range(num_layers):
            ood_layer_output = ood_layer_outputs[f"layer_{layer}"].to(device)

            min_distance = 0
            closest_class = None
            for cls in range(num_classes):
                distance = mahalanobis_distance(ood_layer_output, cls_means[layer][cls], torch.inverse(cls_covariances[layer]))
                if distance < min_distance:
                    min_distance = distance
                    closest_class = cls

            gradient = test_sample.grad
            processed_input_data = test_sample - epsilon * torch.sign(gradient)

            max_score = 0
            for cls in range(num_classes):
                distance = mahalanobis_distance(processed_input_data, cls_means[layer][cls], torch.inverse(cls_covariances[layer]))
                max_score = max(max_score, -distance)

            mds_score.append(alpha_weights[layer] * max_score)

        final_mds_score = sum(mds_score)
        return final_mds_score

# Example usage:
alpha_weights = [1.0] * num_layers  # Define your layer-specific weights
epsilon = 0.01  # Small noise level
test_sample = torch.randn(3, 32, 32).to(device)  # Example test sample in CIFAR-10 format (3 channels, 32x32)
confidence_score = mds_score(model, test_sample, epsilon, cls_means, cls_covariances, num_layers, alpha_weights)
print("Confidence Score:", confidence_score)
