import torch
from datasets import load_dataset
from ood_utils.ood_metrics import evaluations
from models import ResNet, DenseNet
from ood_methods import get_ood_methods
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pre-trained model: resnet18, densenet100
def load_trained_model(model_path, model_type):
    if model_type == "ResNet":
        model = ResNet(3).to(device)
    elif model_type == "DenseNet":
        model = DenseNet(3).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

@torch.no_grad
def run_ood_posthoc_method(args):
    model = args.model  # ResNet, DenseNet
    batch_size = args.batch_size  # 16, 32, 64
    id_dataset = args.id_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    ood_dataset = args.ood_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    method = args.method  # msp, odin, mds, react, logitnorm, knn

    # load model
    if model == "ResNet":  # trained resnet18
        model = load_trained_model("logs/ResNet/trained_model/trained_ResNet_20241211_162024.pth", "ResNet")
    elif model == "ResNet-imported":  # pytorch-pre-trained resnet18 for debugging
        model = models.resnet18(pretrained=True)
    elif model == "DenseNet":  # trained densenet100
        model = load_trained_model("logs/DenseNet/trained_model/trained_DenseNet_20241211_154102.pth", "DenseNet")
    elif model == "DenseNet-imported":  # pytorch-pre-trained densenet100 for debugging
        model = models.densenet121(pretrained=True)
    model.to(device)

    # load id_trainset, id_testset, ood_testset
    id_trainset, id_testset, _, _ = load_dataset(id_dataset)  # trainset, testset, input_channel, image_size
    _, ood_testset, _, _ = load_dataset(ood_dataset)
    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=batch_size, shuffle=True)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    # apply methods using id_trainset
    id_scores = []
    ood_scores = []

    ood_method = get_ood_methods(method, model)
    ood_method.apply_method(id_train_loader)

    # id_scores: get id_testset's scores
    for images, _ in id_test_loader:
        images.to(device)
        batch_id_scores = ood_method.ood_score(images).cpu().numpy()  # (batch)
        id_scores.append(batch_id_scores)

    # ood_scores: get ood_testset's scores
    for images, _ in ood_loader:
        images.to(device)
        batch_ood_scores = ood_method.ood_score(images).cpu().numpy()  # (batch)
        ood_scores.append(batch_ood_scores)

    # get FPR95, AUROC and AUPR
    results = evaluations(id_scores, ood_scores)
    print(results)  # example: {'FPR95': 0.001, 'AUROC': 0.999, 'AUPR': 0.999}
