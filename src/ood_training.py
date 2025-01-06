import torch
import torch.nn.functional as F
from datasets import load_dataset
from models import ResNet, DenseNet, load_model
from ood_methods import get_ood_methods
from ood_utils.ood_metrics import evaluations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ood_training(args):
    # load dataset
    id_dataset = args.id_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    ood_dataset = args.ood_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    id_trainset, id_testset, id_input_channels, id_image_size = load_dataset(id_dataset)
    _, ood_testset, _, _ = load_dataset(ood_dataset)

    # data loader
    batch_size = args.batch_size  # 16, 32, 64
    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=batch_size, shuffle=True)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    # load model
    model = load_model(args.model, id_input_channels, id_image_size)  # ResNet, DenseNet
    model.to(device)

    # training methods config
    method = args.method  # msp, odin, mds, react, knn, logitnorm, oe, moe
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
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # initialize the gradients to '0'
            outputs = model(images)  # Forward pass => softmax not applied
            loss = criterion(outputs, labels)  # average loss "over the batch"
            loss.backward()  # back propagation
            optimizer.step()  # update weights
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print("Training completed")

    torch.save(model.state_dict(), f'ood_logs/trained_{model}_{id_dataset}_{ood_dataset}_{method}.pth')
    print("Model saved")

    # test
    print("Testing...")
    model.eval()
    id_msp_score = []
    ood_msp_score = []

    # id_test_set msp score
    for images, labels in id_test_loader:
        output = model(images)
        batch_id_msp = F.softmax(output)
        id_msp_score.append(batch_id_msp)

    # ood_test_set msp score
    for images, labels in ood_test_loader:
        output = model(images)
        batch_ood_msp = F.softmax(output)
        ood_msp_score.append(batch_ood_msp)

    # compute, return FPR95, AUROC, AUPR
    print("logitnorm evaluation results: ")
    results = evaluations(id_msp_score, ood_msp_score)
    print(results)