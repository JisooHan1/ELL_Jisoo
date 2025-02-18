import torch
import numpy as np
from datasets.datasets import get_data_loaders
from models import load_model, load_saved_model, model_path
from utils.ood_configs import get_training_config
from utils.config import config

# device
def get_device():
    """Get available device (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# initialize training
def initialize_training(device):
    """Initialize training components"""
    # data
    data_loaders, id_input_channels, id_image_size = get_data_loaders(
        id_dataset=config['train']['id_dataset'],
        oe_dataset=config['train']['oe_dataset'],
        ood_dataset=config['train']['ood_dataset'],
        batch_size=config['general']['batch_size'],
        augment=config['general']['augment'],
        csi=config['train']['csi']
    )

    # model
    if config['train']['variant'] is not None:
        model = load_saved_model(
            config['general']['model'],
            model_path[config['general']['model']][config['train']['variant']],
            device
        )
    else:
        model = load_model(config['general']['model'], id_input_channels, id_image_size).to(device)

    # training options
    training_options = get_training_config(config['train']['method'])
    criterion = training_options['criterion']()
    optimizer = training_options['optimizer'](
        model.parameters(),
        lr=training_options['lr'],
        weight_decay=training_options['weight_decay'],
        momentum=training_options['momentum'],
        nesterov=True
    )

    # schedulers
    if training_options['scheduler_type'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=training_options['milestones'],
            gamma=training_options['gamma']
        )
    elif training_options['scheduler_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_options['T_max'],
            eta_min=training_options['eta_min']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {training_options['scheduler_type']}")

    return model, data_loaders, criterion, optimizer, scheduler, training_options['epochs']

# run training
def run_ood_train():
    """Main training function"""
    # Initialize
    device = get_device()
    model, data_loaders, criterion, optimizer, scheduler, epochs = initialize_training(device)
    model.to(device).train()

    print("Start training... method: ", config['train']['method'])
    print("Model: ", config['general']['model'])
    print("Path: ", model_path[config['general']['model']][config['train']['variant']])

    # Training based on method
    if config['train']['method'] == 'logitnorm':
        train_logitnorm(model, data_loaders, criterion, optimizer, scheduler, epochs, device)
    elif config['train']['method'] == 'moe':
        train_moe(model, data_loaders, criterion, optimizer, scheduler, epochs, device)
    elif config['train']['method'] == 'oe':
        train_oe(model, data_loaders, criterion, optimizer, scheduler, epochs, device)
    elif config['train']['method'] == 'csi':
        train_csi(model, data_loaders, criterion, optimizer, scheduler, epochs, device)

# logitnorm
def train_logitnorm(model, data_loaders, criterion, optimizer, scheduler, epochs, device):
    """Train with LogitNorm"""
    for epoch in range(epochs):
        for images, labels in data_loaders['id_train_loader']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("Training completed")
    save_model(model, 'logitnorm')

# moe
def train_moe(model, data_loaders, criterion, optimizer, scheduler, epochs, device):
    """Train with Mixture Outlier Exposure"""
    for epoch in range(epochs):
        for (id_images, id_labels), (oe_images, _) in zip(data_loaders['id_train_loader'], data_loaders['oe_train_loader']):
            id_images, id_labels = id_images.to(device), id_labels.to(device)
            oe_images = oe_images.to(device)

            ratio = np.random.beta(1.0, 1.0)
            mixed_images = ratio * id_images + (1 - ratio) * oe_images

            optimizer.zero_grad()
            id_outputs = model(id_images)
            mixed_outputs = model(mixed_images)
            loss = criterion(id_outputs, id_labels, mixed_outputs, ratio)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("Training completed")
    save_model(model, 'moe')

# oe
def train_oe(model, data_loaders, criterion, optimizer, scheduler, epochs, device):
    """Train with Outlier Exposure"""
    for epoch in range(epochs):
        for (id_images, id_labels), (oe_images, _) in zip(data_loaders['id_train_loader'], data_loaders['oe_train_loader']):
            id_images, id_labels = id_images.to(device), id_labels.to(device)
            oe_images = oe_images.to(device)

            optimizer.zero_grad()
            id_outputs = model(id_images)
            oe_outputs = model(oe_images)
            loss = criterion(id_outputs, id_labels, oe_outputs)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("OE based training completed")
    save_model(model, 'oe')

# csi
def train_csi(model, data_loaders, criterion, optimizer, scheduler, epochs, device):
    """Train with CSI"""
    for epoch in range(epochs):
        for (x11, x12, x21, x22, x31, x32, x41, x42), labels in data_loaders['id_train_loader']:
            inputs = [x.to(device) for x in [x11, x12, x21, x22, x31, x32, x41, x42]]
            labels = labels.to(device)

            batch = torch.cat(inputs, dim=0)
            batch_labels = torch.cat([labels] * 8, dim=0)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("CSI training completed")
    save_model(model, 'csi')

# save model
def save_model(model, method):
    """Save trained model"""
    save_path = f'logs/{config["general"]["model"]}/trained_model/{config["train"]["variant"]}_ood_{method}'

    if method in ['moe', 'oe']:
        save_path += f'_{config["train"]["id_dataset"]}_{config["train"]["oe_dataset"]}.pth'
    else:
        save_path += f'_{config["train"]["id_dataset"]}.pth'

    torch.save(model.state_dict(), save_path)
    print(f"Model saved in {save_path}")

if __name__ == "__main__":
    run_ood_train()