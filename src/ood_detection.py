import torch
import torch.nn.functional as F
from datasets import load_dataset
from models import ResNet
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def load_model(model_path):
    model = ResNet(3)  # input channels
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def get_MSP(input_data, model):

    model.eval()

    with torch.no_grad():
        outputs = model(input_data)
        softmax_scores = F.softmax(outputs, dim=1)
        max_score, _ = torch.max(softmax_scores, dim=1)  # gets MSP

    return max_score.item()

def evaluate_ood_detection(id_scores, ood_scores):
    # generate list of label: ID = 1, OOD = 0
    id_label = [1] * len(id_scores)
    ood_label = [0] * len(ood_scores)
    labels = labels = id_label + ood_label

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


# import torch
# import torch.nn.functional as F
# from datasets import load_dataset
# from models import ResNet
# import argparse

# def load_model(model_path):
#     model = ResNet(3)  # 입력 채널 수가 3인 ResNet 모델 초기화
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval()
#     return model

# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()

#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy:.2f}%')

# def main():
#     parser = argparse.ArgumentParser(description="CIFAR-10 Test Evaluation")
#     parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = load_model(args.model_path)
#     model.to(device)

#     _, testset = load_dataset("CIFAR10")
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

#     evaluate_model(model, test_loader, device)

# if __name__ == "__main__":
#     main()
