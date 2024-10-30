import torch
import torch.nn.functional as F
from datasets import load_dataset
from models import ResNet
import argparse
from torcheval.metrics import BinaryAUROC, BinaryAUPRC

def load_model(model_path):
    # Recover trained model
    model = ResNet(3)
    model.load_state_dict(torch.load(model_path, map_location='cpu')) # pram. recovery
    model.eval()
    return model

def get_MSP(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        softmax_scores = F.softmax(outputs, dim=1)  # (batch_size, num_class)
        max_score, _ = torch.max(softmax_scores, dim=1)  # output: MSP, index - (batch_size)

    return max_score  # type: tensor

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
    parser = argparse.ArgumentParser(description="OOD detection-MSP")
    parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("-id", "--id_dataset", type=str, required=True, help="ID dataset CIFAR10")
    parser.add_argument("-ood", "--ood_dataset", type=str, required=True, help="OOD dataset SVHN")
    args = parser.parse_args()

    model = load_model(args.model_path)
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ID, OOD data load
    _, id_testset = load_dataset(args.id_dataset)
    _, ood_testset = load_dataset(args.ood_dataset)

    # DataLoader, only use test data
    id_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    # get softmax
    id_scores = torch.tensor([], device=device)
    ood_scores = torch.tensor([], device=device)

    for data in id_loader:
        scores = get_MSP(data[0].to(device), model)
        id_scores = torch.cat([id_scores, scores])

    for data in ood_loader:
        scores = get_MSP(data[0].to(device), model)
        ood_scores = torch.cat([ood_scores, scores])

    # get AUROC and AUPR
    evaluate_ood_detection(id_scores, ood_scores)

if __name__ == "__main__":
    main()
