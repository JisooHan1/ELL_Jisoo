import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from models import LeNet, ResNet, DenseNet, FractalNet, ViT, MLPMixer, ConvMixer, load_model, optimizer_and_scheduler
from utils.config import config

from torch.utils.tensorboard import SummaryWriter
import os
import math
from datetime import datetime

def train(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: torch.device) -> None:

    print(f"Starting training for epoch {epoch}")
    model.train()
    running_loss = 0.0 # batches
    total_loss = 0.0 # epochs
    print_frequency = get_print_frequency(len(trainloader))  # batches

    for batch_index, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()               # initialize the gradients to '0'
        outputs = model(images)             # Forward pass => softmax not applied
        loss = criterion(outputs, labels)   # average loss "over the batch"
        loss.backward()                     # back propagation
        optimizer.step()                    # update weights

        # update loss
        running_loss += loss.item() # for batches
        total_loss += loss.item() # for epochs

        # log: loss of every print_frequency batches
        if (batch_index + 1) % print_frequency == 0: # every print_frequency
            avg_loss = running_loss / print_frequency  # avg of avg...
            print(f'[{epoch}, {batch_index + 1}] loss: {avg_loss:.3f}') # average loss for an amount of batches
            writer.add_scalar('Loss/train', avg_loss, epoch * len(trainloader) + batch_index) # global index for loss
            running_loss = 0.0

    # log: total loss for one epoch
    writer.add_scalar('Loss/train_epoch', total_loss / len(trainloader), epoch) # average loss for each epoch

def test(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: torch.device) -> None:

    model.eval()    # dropout X & batch normalization X
    correct = 0     # number of correctly classified images
    total = 0       # number of total images
    test_loss = 0.0 # total loss in test set (batch 단위)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # num of 'True' for pred==label

    accuracy = 100 * correct / total
    print(f'epoch: {epoch}, Accuracy: {accuracy:.2f} %, Avg loss: {test_loss / len(testloader):.3f}')
    writer.add_scalar('Loss/test', test_loss / len(testloader), epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

def main():
    device = get_device()

    # Load config
    model_name = config['general']['model']
    dataset_name = config['general']['dataset']
    augment = config['general']['augment']
    epoch = config['general']['epoch']
    batch_size = config['general']['batch_size']

    # Load data
    trainset, testset, input_channels, image_size = load_dataset(dataset_name, augment)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=2)

    # Load model
    model = load_model(model_name, input_channels, image_size)
    model.to(device)

    # Optimization
    optimizer, scheduler = optimizer_and_scheduler(model, model_name, epoch)
    criterion = nn.CrossEntropyLoss()

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=f'logs/{model_name}/tensorboard_logs')

    # train & test
    for epoch in range(1, epoch + 1):
        train(model, trainloader, criterion, optimizer, epoch, writer, device)
        test(model, testloader, criterion, epoch, writer, device)
        scheduler.step()

    # Compute total number of parameters of the model
    num_of_pars = sum(p.numel() for p in model.parameters())
    print("Finished Training & Testing\n")
    print(f"Number of Parameters in {model_name}: {num_of_pars}")

    # save trained model
    save_model(model, model_name, dataset_name, augment)

# utils
def get_device():
    if torch.cuda.is_available():
        print("GPU Available... Using GPU")
        return torch.device("cuda")
    else:
        print("GPU is not available...Using CPU")
        return torch.device("cpu")

def get_print_frequency(number_of_batches: int) -> int:
    return 200 if number_of_batches > 200 else 15

def save_model(model, model_name, dataset, augment, save_dir="logs"):
    os.makedirs(f'{save_dir}/{model_name}/trained_model', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = f'{save_dir}/{model_name}/trained_model/{model_name}_{dataset}_{augment}_{timestamp}.pth'
    model.to('cpu')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved in {save_path}")
    return save_path

if __name__ == "__main__":
    main()