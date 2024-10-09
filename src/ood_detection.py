import torch
import torch.nn.functional as F
from datasets import load_dataset
from models import ResNet
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def load_model(model_path):
    net = ResNet(3)  # input channels

    # 저장된 모델 가중치 로드
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    return net

def get_MSP(input_data, model):

    model.eval()  # no dropouts

    with torch.no_grad():
        outputs = model(input_data)
        softmax_scores = F.softmax(outputs, dim=1)
        max_score, _ = torch.max(softmax_scores, dim=1)  # gets MSP

    return max_score.item()

def evaluate_ood_detection(id_scores, ood_scores):
    # generates list of label: ID = 0, OOD = 1
    id_label = [0] * len(id_scores)
    ood_label = [1] * len(ood_scores)
    labels = id_label.extend(ood_label)

    scores = id_scores + ood_scores

    auroc = roc_auc_score(labels, scores)  # AUROC

    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)  # AUPR

    print(f'AUROC: {auroc:.4f}')
    print(f'AUPR: {aupr:.4f}')

def main():
    parser = argparse.ArgumentParser(description="OOD detection")
    parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("-id", "--id_dataset", type=str, required=True, help="ID dataset CIFAR10")
    parser.add_argument("-ood", "--ood_dataset", type=str, required=True, help="OOD dataset SVHN")
    args = parser.parse_args()

    model = load_model(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ID, OOD data load
    _, id_testset = load_dataset(args.id_dataset)
    _, ood_testset = load_dataset(args.ood_dataset)

    # DataLoader, only use test data
    id_loader = torch.utils.data.DataLoader(id_testset, batch_size=1, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=1, shuffle=True)

    # get softmax
    id_scores = [get_MSP(data[0].to(device), model) for data in id_loader]
    ood_scores = [get_MSP(data[0].to(device), model) for data in ood_loader]

    # get, AUROC AUPR
    evaluate_ood_detection(id_scores, ood_scores)

if __name__ == "__main__":
    main()
