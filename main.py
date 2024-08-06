import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from models import load_model

import argparse
from torch.utils.tensorboard import SummaryWriter
import sys

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
    print(f"Starting training for epoch {epoch + 1}")
    net.train() # using dropout & batch normalization
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0): # loops over batches
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
        running_loss += loss.item() # extract scalar from tensor
        if i % 2000 == 1999: # print every 2000 mini-batches
            # (current epoch, total batches processed, average loss for last 2000 batches) => avg of avg??
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            writer.add_scalar('Loss/train', running_loss / 2000, epoch * len(trainloader) + i) # global index for loss
            running_loss = 0.0
    writer.add_scalar('Loss/train_epoch', running_loss, epoch)

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
    print('Accuracy of the network on test images: %d %%' % accuracy)
    writer.add_scalar('Loss/test', test_loss / len(testloader), epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

# main
def main():

    device = get_device()

    # Parsing command 정의
    parser = argparse.ArgumentParser(description="Executes deep learning using CCN")
    parser.add_argument("-md", "--model", type=str, required=True, help="type of model: LeNet-5")
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="type of dataset: CIFAR-10, MNIST")
    parser.add_argument("-ep", "--num_epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="batch size")
    args = parser.parse_args()

    # Parsing 변수 정의
    trainset, testset = load_dataset(args.dataset)
    net = load_model(args.model, load_dataset.input_channels, load_dataset.image_size)
    net.to(device)
    epoch = args.num_epochs
    batch_size = args.batch_size

    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir='logs')

    for epoch_index in range(epoch):
        train(net, trainloader, criterion, optimizer, epoch_index, writer, device)
        test(net, testloader, criterion, epoch_index, writer, device)

    print("Finished Training & Testing")

if __name__ == "__main__":
    main()