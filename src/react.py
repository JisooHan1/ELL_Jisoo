import torch
from models import ResNet
from datasets import load_dataset
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Load the model
def load_model(model_path):
    model = ResNet(3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

model = load_model("logs/ResNet/trained_model/trained_resnet_20241009_185530.pth")

# load datasets
id_trainset, id_testset = load_dataset("CIFAR10")
ood_trainset, ood_testset = load_dataset("SVHN")

id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=2)
id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=2)
ood_train_loader = torch.utils.data.DataLoader(ood_trainset, batch_size=64, shuffle=True, num_workers=2)
ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=2)


# 2. Get penultimate layer activations
activations = {}  # {'penultimate': (batch, 512, 1, 1)}

# layer activation hook function
def get_activation(name):
    def hook(_model, _input, output):
        # GAP 이후, FC layer 이전의 512차원 벡터
        activations[name] = output.detach().to(device)
    return hook

# Register hook for penultimate layer (GAP output)
hook = model.GAP.register_forward_hook(get_activation('penultimate'))

def react(x, c=1.0):
    # Ensure input is on the correct device
    x = x.to(device)

    # Forward pass to get activations
    model(x)

    # Get penultimate layer activations
    penultimate = activations['penultimate']  # (batch, 512, 1, 1)
    penultimate = penultimate.flatten(1)      # (batch, 512)

    # Clamp values to be <= c
    clamped = torch.clamp(penultimate, max=c)

    # Get logits and probabilities
    logits = model.fc(clamped)
    probs = F.softmax(logits, dim=1)

    # Get maximum probability as score
    scores = torch.max(probs, dim=1)[0]

    return scores


# Example usage
def evaluate_ood_detection():
    id_scores = []
    ood_scores = []

    with torch.no_grad():
        # Get ID scores
        for images, _ in id_test_loader:
            scores = react(images)
            id_scores.extend(scores.cpu().numpy())

        # Get OOD scores
        for images, _ in ood_test_loader:
            scores = react(images)
            ood_scores.extend(scores.cpu().numpy())

    return id_scores, ood_scores

def test_react():
    print("Testing ReAct implementation...")

    # 1. Test single batch
    for images, _ in id_test_loader:
        scores = react(images)
        print("\nSingle ID batch test:")
        print(f"Batch shape: {images.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        break

    # 2. Test full evaluation
    print("\nFull evaluation test:")
    id_scores, ood_scores = evaluate_ood_detection()

    print(f"Number of ID samples: {len(id_scores)}")
    print(f"Number of OOD samples: {len(ood_scores)}")
    print(f"\nID scores - Mean: {np.mean(id_scores):.4f}, Std: {np.std(id_scores):.4f}")
    print(f"OOD scores - Mean: {np.mean(ood_scores):.4f}, Std: {np.std(ood_scores):.4f}")

if __name__ == "__main__":
    test_react()