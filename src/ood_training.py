import torch
import torch.nn.functional as F
from datasets import load_dataset
from models import ResNet, DenseNet, load_model
from ood_methods import get_ood_methods
from ood_utils.ood_metrics import evaluations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(id_dataset, ood_dataset, batch_size):   # CIFAR10, STL10, CIFAR100, SVHN, LSUN...
    id_trainset, id_testset, id_input_channels, id_image_size = load_dataset(id_dataset)
    _, ood_testset, _, _ = load_dataset(ood_dataset)

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=batch_size, shuffle=True)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)
    return id_train_loader, id_test_loader, ood_test_loader, id_input_channels, id_image_size

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_saved_model(model_path, model):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded from {model_path}")
    return model

# ood training
def ood_training(args):
    # load data
    id_train_loader, id_test_loader, ood_test_loader, id_input_channels, id_image_size = load_data(args.id_dataset, args.ood_dataset, args.batch_size)

    # load model
    model = load_model(args.model, id_input_channels, id_image_size).to(device)  # ResNet, DenseNet

    # option
    # train model
    if args.train.lower() == 'true':
        print("Training...")
        method = args.method  # logitnorm, oe, moe ...

        # training methods config
        if method == 'logitnorm':
            from ood_methods.logitnorm import LogitNormLoss
            criterion = LogitNormLoss(tau=0.04)
            lr = 0.1
            weight_decay = 5e-4
            momentum = 0.9
            epochs = 200
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140], gamma=0.1)

        elif method == 'oe':
            from ood_methods.outlier_exposure import OutlierExposureLoss
            criterion = OutlierExposureLoss(alpha=0.5)

        elif method == 'moe':
            from ood_methods.mixture_outlier_exposure import MixtureOutlierExposureLoss
            criterion = MixtureOutlierExposureLoss(alpha=0.5, beta=0.5)

        # train
        for epoch in range(epochs):
            model.train()
            for images, labels in id_train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()  # initialize the gradients to '0'
                outputs = model(images)  # Forward pass => softmax not applied
                loss = criterion(outputs, labels)  # average loss "over the batch"
                loss.backward()  # back propagation
                optimizer.step()  # update weights
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        print("Training completed")
        save_model(model, f'logs/{args.model}/trained_model/ood_{method}_{args.id_dataset}_{args.ood_dataset}.pth')

    # load trained model
    else:
        model = load_saved_model(f'logs/{args.model}/trained_model/ood_{args.method}_{args.id_dataset}_{args.ood_dataset}.pth', model)

    return model, id_test_loader, ood_test_loader

def evaluate(model, id_test_loader, ood_test_loader):
    # test
    print("Testing...")
    model.eval()
    id_msp_score = []
    ood_msp_score = []

    # id_test_set msp score
    with torch.no_grad():
        for images, _ in id_test_loader:
            images = images.to(device)
            output = model(images)
            batch_id_msp = F.softmax(output)
            id_msp_score.append(batch_id_msp)

    # ood_test_set msp score
    with torch.no_grad():
        for images, _ in ood_test_loader:
            images = images.to(device)
            output = model(images)
            batch_ood_msp = F.softmax(output)
            ood_msp_score.append(batch_ood_msp)

    # compute, return FPR95, AUROC, AUPR
    print("evaluation results: ")
    results = evaluations(id_msp_score, ood_msp_score)
    print(results)
    return results