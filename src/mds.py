import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from models import ResNet, DenseNet
import argparse
from torcheval.metrics import BinaryAUROC, BinaryAUPRC
from ood_methods import get_ood_methods


# definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id_layer_outputs = {}  # {'layer1': (batch, channel), 'layer2': (batch, channel), ...}
ood_layer_outputs = {}  # {'layer1': (batch, channel), 'layer2': (batch, channel), ...}

num_classes = 10
num_layers = 50  # 48 DenseBlock, 2 TransitionBlock

class_features = {layer: {cls: [] for cls in range(num_classes)} for layer in range(num_layers)}  # {layer: {cls: [(channel), ...]}}
cls_means = {layer: {cls: None for cls in range(num_classes)} for layer in range(num_layers)}  # {layer: {cls: mean (channel)}}
cls_covariances = {layer: None for layer in range(num_layers)}  # {layer: covariance (channel, channel)}

avg_pool = nn.AdaptiveAvgPool2d((1, 1))


# load model
def load_model(model_path):
    model = DenseNet(3).to(device)

    # load pretrained weights to model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # set model to evaluation mode
    model.eval()

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


if __name__ == '__main__':
    # load pretrained model
    model = load_model("logs/DenseNet/trained_model/trained_resnet_20241031_004039.pth")

    # load datasets
    id_trainset, id_testset = load_dataset("CIFAR10")
    ood_trainset, ood_testset = load_dataset("SVHN")

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=0)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=0)
    ood_train_loader = torch.utils.data.DataLoader(ood_trainset, batch_size=64, shuffle=True, num_workers=0)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=0)

    # Obtain id_layer_outputs dictionary
    # layer activation hook function
    def get_activation(layer_name, output_dict):
        def hook(_model, _input, output):
            # (batch, channel, h, w) -> (batch, channel, 1, 1) -> (batch, channel)
            pooled_output = avg_pool(output).squeeze().to(device)

            output_dict[layer_name] = pooled_output.detach().to(device)  # Detach to save memory
        return hook  # return hook 'function'

    # register hooks for all layers
    hooks = []
    for i in range(num_layers):
        hook = model.dense_layers[i].register_forward_hook(
            get_activation(f"layer_{i}", id_layer_outputs)
        )
        hooks.append(hook)

    # Compute class means and covariances
    # forward pass
    for inputs, labels in id_train_loader:  # (64, 3, 32, 32), (64)
        inputs, labels = inputs.to(device), labels.to(device)
        model(inputs)  # get id_layer_outputs

        # obtain class_features dictionary
        for layer_index, (layer_name, output) in enumerate(id_layer_outputs.items()):
            for i, label in enumerate(labels):
                class_index = label.item()
                class_features[layer_index][class_index].append(output[i].to(device))

        # obtain cls_means and cls_covariances dictionaries
        for layer in range(num_layers):
            layer_data = []
            for cls in range(num_classes):
                class_data = torch.stack(class_features[layer][cls], dim=0).to(device)  # (n_samples, channel)
                cls_means[layer][cls] = torch.mean(class_data, dim=0)  # (channel)

                layer_data.append(class_data)
            layer_data = torch.cat(layer_data, dim=0).to(device)  # (total_samples, channel)
            layer_mean = torch.mean(layer_data, dim=0).to(device)  # (channel)

            N = layer_data.shape[0]  # Total number of samples
            deviations = layer_data - layer_mean.unsqueeze(0)  # (total_samples, channel)
            covariance = torch.matmul(deviations.T, deviations) / (N - 1)  # (channel, channel)
            cls_covariances[layer] = covariance

    # ... existing code ...

    # Print debug information
    print("\nDebug Information:")

    # 1. Check layer outputs shape
    print("\nLayer Outputs Shape:")
    for layer_name, output in id_layer_outputs.items():
        print(f"{layer_name}: {output.shape}")

    # 2. Check class means for first layer
    print("\nClass Means (Layer 0):")
    for cls in range(num_classes):
        print(f"Class {cls}: Mean shape = {cls_means[0][cls].shape}")
        print(f"First few values: {cls_means[0][cls][:5]}")

    # 3. Check covariance matrix for first layer
    print("\nCovariance Matrix (Layer 0):")
    print(f"Shape: {cls_covariances[0].shape}")
    print("First 5x5 values:")
    print(cls_covariances[0][:5, :5])

    # 4. Check number of samples per class in first layer
    print("\nSamples per class (Layer 0):")
    for cls in range(num_classes):
        print(f"Class {cls}: {len(class_features[0][cls])} samples")