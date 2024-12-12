import torch
import numpy as np
from datasets import load_dataset
from models import ResNet, DenseNet
import argparse
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from ood_methods import get_ood_methods

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loading function
def load_trained_model(model_path, model_type):
    if model_type == "ResNet":
        model = ResNet(3).to(device)
    elif model_type == "DenseNet":
        model = DenseNet(3).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def evaluate_ood_detection(id_scores, ood_scores):
    id_scores = id_scores.cpu().numpy()
    ood_scores = ood_scores.cpu().numpy()

    # generate list of label: ID = 1, OOD = 0
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])  # (num_samples, 1)

    # FPR95
    fprs, tprs, _ = roc_curve(labels, scores)
    tpr_95_idx = np.argmin(np.abs(tprs - 0.95))
    fpr95 = fprs[tpr_95_idx]

    # AUROC, AUPRC
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    print(f'FPR95: {fpr95:.4f}')
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPR: {aupr:.4f}')


def run_ood_detection(args):
    # load model
    if args.model == "ResNet":
        model = load_trained_model("logs/ResNet/trained_model/trained_ResNet_20241211_162024.pth", "ResNet")
    elif args.model == "DenseNet":
        model = load_trained_model("logs/DenseNet/trained_model/trained_DenseNet_20241211_154102.pth", "DenseNet")
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
    id_scores = []
    ood_scores = []

    ood_method = get_ood_methods(args.method, model)

    if args.method == "mds":
        class_features = ood_method.get_class_features(id_loader)
        ood_method.get_cls_means(class_features)
        ood_method.get_cls_covariances(class_features)
        score_func = ood_method.mds_score
    elif args.method == "react":
        ood_method.get_samples(id_loader)
        ood_method.calculate_threshold(ood_method.samples)
        score_func = ood_method.react_score
    elif args.method == "knn":
        ood_method.get_features(id_loader)
        score_func = ood_method.knn_score
    else:
        score_func = ood_method

    for data in id_loader:
        scores = score_func(data[0].to(device), model)
        id_scores.append(scores)
    id_scores = torch.cat(id_scores)

    for data in ood_loader:
        scores = score_func(data[0].to(device), model)
        ood_scores.append(scores)
    ood_scores = torch.cat(ood_scores)

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
