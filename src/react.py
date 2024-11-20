# import torch
# from models import ResNet
# from datasets import load_dataset
# import torch.nn.functional as F
# import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # 1. Load the model
# def load_model(model_path):
#     model = ResNet(3).to(device)
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#     return model

# model = load_model("logs/ResNet/trained_model/trained_resnet_20241009_185530.pth")

# # load datasets
# id_trainset, id_testset = load_dataset("CIFAR10")
# ood_trainset, ood_testset = load_dataset("SVHN")

# id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=2)
# id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=2)
# ood_train_loader = torch.utils.data.DataLoader(ood_trainset, batch_size=64, shuffle=True, num_workers=2)
# ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=2)


# # 2. Get penultimate layer activations
# activations = {}  # {'penultimate': (batch, 512, 1, 1)}

# # layer activation hook function
# def get_activation(name):
#     def hook(_model, _input, output):
#         # GAP 이후, FC layer 이전의 512차원 벡터
#         activations[name] = output.detach().to(device)
#     return hook

# # Register hook for penultimate layer (GAP output)
# hook = model.GAP.register_forward_hook(get_activation('penultimate'))

# def react(x, c=1.0):
#     # Ensure input is on the correct device
#     x = x.to(device)

#     # Forward pass to get activations
#     model(x)

#     # Get penultimate layer activations
#     penultimate = activations['penultimate']  # (batch, 512, 1, 1)
#     penultimate = penultimate.flatten(1)      # (batch, 512)

#     # Calculate threshold c using mean + k*std of ID activations
#     # This is another way to set threshold covering majority of ID data
#     means = penultimate.mean(dim=0)  # (512,)
#     stds = penultimate.std(dim=0)    # (512,)
#     k = 1.28  # Covers ~90% of data assuming normal distribution
#     c = (means + k * stds).to(device)

#     # Clamp values to be <= c
#     clamped = torch.clamp(penultimate, max=c.unsqueeze(0))  # broadcast c to match batch dimension

#     # Get logits and probabilities
#     logits = model.fc(clamped)
#     probs = F.softmax(logits, dim=1)

#     # Get maximum probability as score
#     scores = torch.max(probs, dim=1)[0]

#     return scores


# # Example usage
# def evaluate_ood_detection():
#     id_scores = []
#     ood_scores = []

#     with torch.no_grad():
#         # Get ID scores
#         for images, _ in id_test_loader:
#             scores = react(images)
#             id_scores.extend(scores.cpu().numpy())

#         # Get OOD scores
#         for images, _ in ood_test_loader:
#             scores = react(images)
#             ood_scores.extend(scores.cpu().numpy())

#     return id_scores, ood_scores

# def test_react():
#     print("Testing ReAct implementation...")

#     # 1. Test single batch
#     for images, _ in id_test_loader:
#         scores = react(images)
#         print("\nSingle ID batch test:")
#         print(f"Batch shape: {images.shape}")
#         print(f"Scores shape: {scores.shape}")
#         print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
#         break

#     # 2. Test full evaluation
#     print("\nFull evaluation test:")
#     id_scores, ood_scores = evaluate_ood_detection()

#     print(f"Number of ID samples: {len(id_scores)}")
#     print(f"Number of OOD samples: {len(ood_scores)}")
#     print(f"\nID scores - Mean: {np.mean(id_scores):.4f}, Std: {np.std(id_scores):.4f}")
#     print(f"OOD scores - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}")

# if __name__ == "__main__":
#     test_react()

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

class ReActResNet(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        # ResNet50 백본
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.threshold = threshold

    def react(self, x):
        return torch.clamp(x, max=self.threshold)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.react(self.backbone.layer1(x))
        x = self.react(self.backbone.layer2(x))
        x = self.react(self.backbone.layer3(x))
        x = self.react(self.backbone.layer4(x))

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x

    def get_scores(self, x):
        with torch.no_grad():
            out = self.forward(x)
            scores = torch.softmax(out, dim=1).max(dim=1)[0]
        return scores.cpu().numpy()

def evaluate_ood(model, id_loader, ood_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ID 점수 수집
    id_scores = []
    for x, _ in id_loader:
        scores = model.get_scores(x.to(device))
        id_scores.extend(scores)

    # OOD 점수 수집
    ood_scores = []
    for x, _ in ood_loader:
        scores = model.get_scores(x.to(device))
        ood_scores.extend(scores)

    # 메트릭 계산
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)

    # FPR95
    threshold = np.percentile(id_scores, 5)
    fpr95 = np.sum(ood_scores >= threshold) / len(ood_scores)

    # AUROC
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_scores)

    return fpr95, auroc

# 데이터 로더 설정
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 예시 실행
if __name__ == "__main__":
    model = ReActResNet(threshold=0.5)

    # 데이터셋 로드 (예시)
    imagenet_val = torchvision.datasets.ImageFolder('path/to/imagenet/val', transform=transform)
    ood_dataset = torchvision.datasets.ImageFolder('path/to/ood/dataset', transform=transform)

    id_loader = DataLoader(imagenet_val, batch_size=64, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False)

    # OOD 평가
    fpr95, auroc = evaluate_ood(model, id_loader, ood_loader)
    print(f"FPR95: {fpr95:.2f}, AUROC: {auroc:.2f}")