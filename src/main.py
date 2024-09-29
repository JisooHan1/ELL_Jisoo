import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from models import LeNet, ResNet, DenseNet, FractalNet, load_model

import argparse
from torch.utils.tensorboard import SummaryWriter
import sys
import math

# check gpu
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU Available... Using GPU")
    else:
        print("GPU is not available...\n")
        user_input = input("'1': CPU / Else: Exit\n")
        if user_input == "1":
            print("Using CPU")
            device = torch.device("cpu")
        else:
            print("Exiting")
            sys.exit()
    return device


# training on one epoch
def train(net, trainloader, criterion, optimizer, epoch, writer, device):
    print(f"Starting training for epoch {epoch}")
    net.train() # using dropout & batch normalization
    running_loss = 0.0 # for batches
    total_loss = 0.0 # for epochs
    number_of_batches = len(trainloader)

    # print batch manage
    if number_of_batches > 200:
        print_frequency = 200
    else:
        print_frequency = 15

    # iterates over elements of  trainloader (one batch). 0: starting batch index of 'i'
    for batch_index, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad() # initialize the gradients to '0'
        outputs = net(inputs) # Forward pass => softmax not applied
        loss = criterion(outputs, labels) # average loss "over the batch"
        loss.backward() # back propagation
        optimizer.step() # update weights

        # print statistics
        running_loss += loss.item() # extract scalar from tensor: for batches
        total_loss += loss.item() # extract scalar from tensor: for epochs
        if (batch_index+1) % print_frequency == 0: # print every (print_frequency) mini-batches
            # (current epoch, total batches processed, average loss for last (print_frequency) batches) => avg of avg??
            print('[%d, %5d] loss: %.3f' %(epoch, batch_index + 1, running_loss / print_frequency)) # average loss for an amount of batches
            writer.add_scalar('Loss/train', running_loss / print_frequency, epoch * len(trainloader) + batch_index) # global index for loss
            running_loss = 0.0
    writer.add_scalar('Loss/train_epoch', total_loss / len(trainloader), epoch) # average loss for each epoch

# testing on one epoch
def test(net, testloader, criterion, epoch, writer, device):
    net.eval() # not using dropout & batch normalization
    correct = 0 # number of correctly classified images
    total = 0 # number of total images
    test_loss = 0.0 # total loss in test set (batch 단위)

    with torch.no_grad(): # doesn't compute gradients while testing
        for data in testloader: # loops over each batches
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels) # average loss "over the batch"
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1) # extract higest valued class (batch 단위)
            total += labels.size(0) # 결국 batch size
            correct += (predicted == labels).sum().item() # num of 'True' for pred==label

    accuracy = 100 * correct / total
    print('epoch: %d, Accuracy of the network on test images: %d %%' %(epoch, accuracy))
    writer.add_scalar('Loss/test', test_loss / len(testloader), epoch) # average loss for one batch in total dataset
    writer.add_scalar('Accuracy/test', accuracy, epoch)

# main
def main():

    device = get_device()

    # Parsing command 정의
    parser = argparse.ArgumentParser(description="Executes deep learning using CCN")
    parser.add_argument("-md", "--model", type=str, required=True, help="type of model: LeNet5, ResNet18, DensNet100, FractalNet")
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="type of dataset: CIFAR-10, MNIST, STL-10")
    parser.add_argument("-ep", "--num_epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="batch size")
    args = parser.parse_args()

    # Parsing 변수 정의
    trainset, testset = load_dataset(args.dataset)
    net = load_model(args.model, load_dataset.input_channels, load_dataset.image_size)
    net.to(device)
    epoch = args.num_epochs
    batch_size = args.batch_size

    # Data Loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=2)

    # Optimization
    if args.model == "LeNet": # batch size: 64, epoch: 300
        lr = 0.001
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        milestones = [epoch*0.5, epoch*0.75]
    elif args.model == "ResNet": # batch size: 64, epoch: 300
        lr = 0.001
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        milestones = [epoch*0.5, epoch*0.75]
    elif args.model == "DenseNet": # batch size: 64, epoch: 300
        lr = 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        milestones = [epoch*0.5, epoch*0.75]
    elif args.model == "FractalNet": # batch size: 64, epoch: 100
        lr = 0.001
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        milestones = [epoch // 2**i for i in range(1, int(math.log2(epoch)) + 1)]
        milestones.reverse()
        # lr = 0.02
        # optimizer = optim.Adam(net.parameters(), lr=lr)
        # # milestones = [epoch*0.5, epoch*0.75]
        # milestones = [epoch // 2**i for i in range(1, int(math.log2(epoch)) + 1)]
        # milestones.reverse()
    elif args.model == "ViT": # batch size: 64, epoch: 100
        lr = 0.001
        optimizer = optim.Adam(net.parameters(), lr=lr)
        milestones = []
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=f'logs/{args.model}')

    # train & test
    for epoch in range(1, epoch+1):
        train(net, trainloader, criterion, optimizer, epoch, writer, device)
        test(net, testloader, criterion, epoch, writer, device)

        scheduler.step()

    # Compute total number of parameters of the model
    num_of_pars = sum(p.numel() for p in net.parameters())


    print("Finished Training & Testing\n")
    print(f"Number of Parameters in {args.model}: {num_of_pars}")

if __name__ == "__main__":
    main()