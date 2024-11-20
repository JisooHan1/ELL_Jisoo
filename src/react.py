# import torch
# from models import ResNet
# from datasets import load_dataset
# import torch.nn.functional as F
# import numpy as np
# from torcheval.metrics import BinaryAUROC, BinaryAUPRC

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # load model
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


# # get penultimate layer activations
# activations = {}  # {'penultimate': (batch, 512, 1, 1)}

# def get_activation(name):
#     def hook(_model, _input, output):
#         activations[name] = output.detach().to(device)
#     return hook

# hook = model.GAP.register_forward_hook(get_activation('penultimate'))


# # Calculate threshold c
# penultimate = activations['penultimate'].flatten(1)  # (batch, 512)

# means = penultimate.mean(dim=0)  # (512,)
# stds = penultimate.std(dim=0)  # (512,)
# k = 1.28  # Covers ~90% of data assuming normal distribution
# c = (means + k * stds).to(device)  # (512,)


# def react(dataloader, penultimate, c):
#     for x, _ in dataloader:
#         x = x.to(device)
#         model(x)

#     clamped = torch.clamp(penultimate, max=c.unsqueeze(0))  # broadcast c to match batch dimension

#     logits = model.fc(clamped)
#     softmax = F.softmax(logits, dim=1)

#     # MSP score
#     scores = torch.max(softmax, dim=1)[0]
#     return scores


# def evaluate_ood_detection(id_scores, ood_scores):
#     # generate list of label: ID = 1, OOD = 0
#     labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)])
#     scores = torch.cat([id_scores, ood_scores])

#     # Use Binary metrics for OOD detection
#     binary_auroc = BinaryAUROC().to(device)
#     binary_auroc.update(scores, labels)
#     binary_auprc = BinaryAUPRC().to(device)
#     binary_auprc.update(scores, labels)

#     auroc = binary_auroc.compute()
#     aupr = binary_auprc.compute()

#     print(f'AUROC: {auroc:.4f}')
#     print(f'AUPR: {aupr:.4f}')

# def run_ood_detection():
#     model.to(device)

#     # get softmax
#     id_scores = react(id_test_loader, penultimate, c)
#     ood_scores = react(ood_test_loader, penultimate, c)

#     # get AUROC and AUPR
#     evaluate_ood_detection(id_scores, ood_scores)



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
import torch.nn.functional as F
from models import ResNet
from datasets import load_dataset
from torcheval.metrics import BinaryAUROC, BinaryAUPRC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReActDetector:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.activations = {}
        self._register_hooks()
        self.c = None

    def _load_model(self, model_path):
        model = ResNet(3).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _get_activation(self, name):
        def hook(_model, _input, output):
            self.activations[name] = output.detach().to(device)
        return hook

    def _register_hooks(self):
        self.model.GAP.register_forward_hook(self._get_activation('penultimate'))

    def calculate_threshold(self, dataloader):
        # Get penultimate activations for ID data
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                self.model(x)

        penultimate = self.activations['penultimate'].flatten(1)
        means = penultimate.mean(dim=0)
        stds = penultimate.std(dim=0)
        k = 1.28  # Covers ~90% of data assuming normal distribution
        self.c = (means + k * stds).to(device)

    def react(self, dataloader):
        scores_list = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                self.model(x)
                penultimate = self.activations['penultimate'].flatten(1)
                clamped = torch.clamp(penultimate, max=self.c.unsqueeze(0))
                logits = self.model.fc(clamped)
                softmax = F.softmax(logits, dim=1)
                scores = torch.max(softmax, dim=1)[0]
                scores_list.append(scores)

        return torch.cat(scores_list)

    def evaluate_ood_detection(self, id_scores, ood_scores):
        labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)])
        scores = torch.cat([id_scores, ood_scores])

        binary_auroc = BinaryAUROC().to(device)
        binary_auroc.update(scores, labels)
        binary_auprc = BinaryAUPRC().to(device)
        binary_auprc.update(scores, labels)

        auroc = binary_auroc.compute()
        aupr = binary_auprc.compute()

        return auroc, aupr

def main():
    # Load datasets
    id_trainset, id_testset = load_dataset("CIFAR10")
    ood_trainset, ood_testset = load_dataset("SVHN")

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=2)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=2)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize ReAct detector
    model_path = "logs/ResNet/trained_model/trained_resnet_20241009_185530.pth"
    detector = ReActDetector(model_path)

    # Calculate threshold using ID training data
    print("Calculating threshold...")
    detector.calculate_threshold(id_train_loader)

    # Evaluate OOD detection
    print("Evaluating OOD detection...")
    id_scores = detector.react(id_test_loader)
    ood_scores = detector.react(ood_test_loader)

    auroc, aupr = detector.evaluate_ood_detection(id_scores, ood_scores)
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPR: {aupr:.4f}')

if __name__ == "__main__":
    main()