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

# Model loading function
def load_pretrained_model(model_path):
    model = DenseNet(3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

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

def run_ood_detection(args):
    # load model
    if args.model == "ResNet":
        model = load_pretrained_model("logs/ResNet/trained_model/trained_resnet_20241009_185530.pth")
    elif args.model == "DenseNet":
        model = load_pretrained_model("logs/DenseNet/trained_model/trained_resnet_20241031_004039.pth")
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load ID, OOD data
    _, id_testset = load_dataset(args.id_dataset)
    _, ood_testset = load_dataset(args.ood_dataset)

    # DataLoader - test data
    id_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    # get softmax
    id_scores = torch.tensor([], device=device)
    ood_scores = torch.tensor([], device=device)

    ood_method = get_ood_methods(args.method)

    for data in id_loader:
        scores = ood_method(data[0].to(device), model)
        id_scores = torch.cat([id_scores, scores])

    for data in ood_loader:
        scores = ood_method(data[0].to(device), model)
        ood_scores = torch.cat([ood_scores, scores])

    # get AUROC and AUPR
    evaluate_ood_detection(id_scores, ood_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOD detection")
    parser.add_argument("-md", "--model", type=str, required=True, help="model")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("-id", "--id_dataset", type=str, required=True, help="ID dataset CIFAR10")
    parser.add_argument("-ood", "--ood_dataset", type=str, required=True, help="OOD dataset SVHN")
    parser.add_argument("-m", "--method", type=str, help="OOD method to use")
    args = parser.parse_args()

    run_ood_detection(args)
