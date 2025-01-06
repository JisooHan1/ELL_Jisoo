import torch
from datetime import datetime
from datasets import load_dataset
from models import ResNet, DenseNet, load_model
from ood_methods import get_ood_methods


def ood_training(args):
    # common arguments
    model = args.model  # ResNet, DenseNet
    batch_size = args.batch_size  # 16, 32, 64
    id_dataset = args.id_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize
    ood_dataset = args.ood_dataset  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, LSUN-resize, ImageNet, ImageNet-resize

    method = args.method  # msp, odin, mds, react, logitnorm, knn
    if method == 'logitnorm':
        from ood_methods.logitnorm import LogitNormLoss
        criterion = LogitNormLoss(tau=tau)
    elif method == 'oe':
        from ood_methods.oe import OutlierExposureLoss
        criterion = OutlierExposureLoss(alpha=0.5)
    elif method == 'moe':
        from ood_methods.mixtureoe import MixtureOutlierExposureLoss
        criterion = MixtureOutlierExposureLoss(alpha=0.5, beta=0.5)

    # training methods arguments
    epoch = args.epoch  # 100
    lr = args.lr  # 0.001
    milestones = args.milestones  # [50, 75]

    # logitnorm arguments
    tau = args.tau  # 0.04

    # load dataset
    id_trainset, id_testset = load_dataset(id_dataset)
    _, ood_testset = load_dataset(ood_dataset)
    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=batch_size, shuffle=True)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

    # load model
    model = load_model(model, load_dataset.input_channels, load_dataset.image_size)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoch):
        model.train()
        for images, labels in id_train_loader:
            optimizer.zero_grad()  # initialize the gradients to '0'
            outputs = model(images)  # Forward pass => softmax not applied
            loss = criterion(outputs, labels)  # average loss "over the batch"
            loss.backward()  # back propagation
            optimizer.step()  # update weights

    torch.save(model.state_dict(), f'ood_logs/trained_{model}_{id_dataset}_{ood_dataset}_{method}.pth')

    # test
    model.eval()

    # id_test_set msp score
    for images, labels in id_test_loader:
        output = model(images)


    # ood_test_set msp score
    for images, labels in ood_test_loader:
        output = model(images;)

    # generate label, score of (id + ood) test data

    # compute, return evalutaion metrics