import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from models import load_model

import argparse
from torch.utils.tensorboard import SummaryWriter
import sys

# check gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Available... Using GPU")
else:
    print("GPU is not available... Using CPU")
    device = torch.device("cpu")


# train
def train(net, trainloader, criterion, optimizer, num_epoch, writer, device):
    net.train() # using dropout & batch normalization
    for epoch in range(num_epoch):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                writer.add_scalar('Loss/train', running_loss / 2000, epoch * len(trainloader) + i) # global index for loss
                running_loss = 0.0

        writer.add_scalar('Loss/train_epoch', running_loss, epoch)
    print('Finished Training')

# train
def test(net, testloader, criterion, num_epoch, writer, device):
    net.eval() # not using dropout & batch normalization
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
    writer.add_scalar('Loss/test', test_loss / len(testloader), num_epoch)
    writer.add_scalar('Accuracy/test', accuracy, num_epoch)

# main
def main():
    # Parsing command 정의
    parser = argparse.ArgumentParser(description="Executes deep learning using CCN")
    parser.add_argument("-md", "--model", type=str, required=True, help="type of model: LeNet-5")
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="type of dataset: CIFAR-10, MNIST")
    parser.add_argument("-ep", "--num_epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="batch size")
    args = parser.parse_args()

    # Parsing 변수 정의
    trainset, testset = load_dataset(args.dataset)
    net = load_model(args.model, load_dataset.input_channels)
    net.to(device)
    num_epoch = args.num_epochs
    batch_size = args.batch_size

    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir='logs')
    train(net, trainloader, criterion, optimizer, num_epoch, writer, device)
    test(net, testloader, criterion, num_epoch, writer, device)

if __name__ == "__main__":
    main()