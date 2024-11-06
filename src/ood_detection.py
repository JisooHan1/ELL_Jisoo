import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from models import ResNet, DenseNet
import argparse
from torcheval.metrics import BinaryAUROC, BinaryAUPRC
from ood_methods import get_ood_methods

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
    model = DenseNet(3)
    state_dict = torch.load(model_path, map_location='cpu')
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
        model(inputs)
        for layer_index, (layer_name, output) in enumerate(id_layer_outputs.items()):
            for i, label in enumerate(labels):
                class_index = label.item()
                class_features[layer_index][class_index].append(output[i])

    for layer in range(num_layers):
        class_deviation_sums = []
        N = sum(len(class_features[layer][cls]) for cls in range(num_classes))

        for cls in range(num_classes):
            class_data = torch.cat(class_features[layer][cls], dim=0)
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

        for layer in range(num_layers):
            ood_layer_output = ood_layer_outputs[f"layer_{layer}"]

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
test_sample = torch.randn(3, 32, 32)  # Example test sample in CIFAR-10 format (3 channels, 32x32)
confidence_score = mds_score(model, test_sample, epsilon, cls_means, cls_covariances, num_layers, alpha_weights)
print("Confidence Score:", confidence_score)



# def evaluate_ood_detection(id_scores, ood_scores):
#     # generate list of label: ID = 1, OOD = 0
#     labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)])
#     scores = torch.cat([id_scores, ood_scores])

#     # Use Binary metrics for OOD detection
#     binary_auroc = BinaryAUROC()
#     binary_auroc.update(scores, labels)
#     binary_auprc = BinaryAUPRC()
#     binary_auprc.update(scores, labels)

#     auroc = binary_auroc.compute()
#     aupr = binary_auprc.compute()

#     print(f'AUROC: {auroc:.4f}')
#     print(f'AUPR: {aupr:.4f}')

# def run_ood_detection(args):
#     # load model
#     model = load_model(args.model_path)
#     batch_size = args.batch_size
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # load ID, OOD data
#     _, id_testset = load_dataset(args.id_dataset)
#     _, ood_testset = load_dataset(args.ood_dataset)

#     # DataLoader - test data
#     id_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
#     ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

#     # get softmax
#     id_scores = torch.tensor([], device=device)
#     ood_scores = torch.tensor([], device=device)

#     ood_method = get_ood_methods(args.method)

#     for data in id_loader:
#         scores = ood_method(data[0].to(device), model)
#         id_scores = torch.cat([id_scores, scores])

#     for data in ood_loader:
#         scores = ood_method(data[0].to(device), model)
#         ood_scores = torch.cat([ood_scores, scores])

#     # get AUROC and AUPR
#     evaluate_ood_detection(id_scores, ood_scores)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="OOD detection")
#     parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
#     parser.add_argument("-bs", "--batch_size", type=int, required=True, help="Batch size")
#     parser.add_argument("-id", "--id_dataset", type=str, required=True, help="ID dataset CIFAR10")
#     parser.add_argument("-ood", "--ood_dataset", type=str, required=True, help="OOD dataset SVHN")
#     parser.add_argument("-m", "--method", type=str, help="OOD method to use")
#     args = parser.parse_args()

#     run_ood_detection(args)
