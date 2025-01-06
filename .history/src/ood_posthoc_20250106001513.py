import torch
from datasets import load_dataset
from ood_utils.ood_metrics import evaluations
from models import ResNet, DenseNet
from ood_methods import get_ood_methods

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def ood_posthoc(args):
    model = args.model  # ResNet, DenseNet
    batch_size = args.batch_size  # 16, 32, 64
    id_dataset = args.id_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    ood_dataset = args.ood_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    method = args.method  # msp, odin, mds, react, logitnorm, knn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if model == "ResNet":
        model = load_trained_model("logs/ResNet/trained_model/trained_ResNet_20241211_162024.pth", "ResNet")
    elif model == "DenseNet":
        model = load_trained_model("logs/DenseNet/trained_model/trained_DenseNet_20241211_154102.pth", "DenseNet")
    model.to(device)

    # load ID, OOD data
    _, id_testset = load_dataset(id_dataset)
    _, ood_testset = load_dataset(ood_dataset)
    id_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    # get softmax
    id_scores = []
    ood_scores = []
    ood_method = get_ood_methods(method, model)
    ood_method.apply_method(id_loader)

    for inputs, _ in id_loader:
        batch_id_scores = ood_method.ood_score(inputs.to(device))
        id_scores.append(batch_id_scores)

    for inputs, _ in ood_loader:
        batch_ood_scores = ood_method.ood_score(inputs.to(device))
        ood_scores.append(batch_ood_scores)

    # get FPR95, AUROC and AUPR
    results = evaluations(id_scores, ood_scores)
    print(results)  # example: {'FPR95': 0.001, 'AUROC': 0.999, 'AUPR': 0.999}
