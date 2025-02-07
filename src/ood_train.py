import torch
from datasets import get_data_loaders
from models import load_model, load_saved_model, model_path
from utils.ood_configs import get_training_config
from utils.config import config

# device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# hyperparameters
def initialize_training(device):
    data_loaders, id_input_channels, id_image_size = get_data_loaders(config['train']['id_dataset'],
                                                                      config['train']['oe_dataset'],
                                                                      config['train']['ood_dataset'],
                                                                      config['general']['batch_size'],
                                                                      config['general']['augment'])

    if config['train']['variant'] is not None:
        model = load_saved_model(config['general']['model'], model_path[config['general']['model']][config['train']['variant']], device)
    else:
        model = load_model(config['general']['model'], id_input_channels, id_image_size).to(device)

    training_options = get_training_config(config['train']['method'])
    criterion = training_options['criterion']()
    optimizer = training_options['optimizer'](
        model.parameters(), lr=training_options['lr'], weight_decay=training_options['weight_decay'], momentum=training_options['momentum'], nesterov=True)

    # schedulers
    if training_options['scheduler_type'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=training_options['milestones'], gamma=training_options['gamma'])
    elif training_options['scheduler_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_options['T_max'], eta_min=training_options['eta_min'])
    else:
        raise ValueError(f"Unsupported scheduler type: {training_options['scheduler_type']}")

    return model, data_loaders, criterion, optimizer, scheduler, training_options['epochs']

# training
def run_ood_train():
    device = get_device()
    model, data_loaders, criterion, optimizer, scheduler, epochs = initialize_training(device)
    model.to(device).train()
    print("Start training... method: ", config['train']['method'])
    print("Model: ", config['general']['model'])
    print("Path: ", model_path[config['general']['model']][config['train']['variant']])

    # Only ID_trainset
    if config['train']['oe_dataset'] is None:
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

        save_path = f'logs/{config["general"]["model"]}/trained_model/{config["train"]["variant"]}_ood_{config["train"]["method"]}_{config["train"]["id_dataset"]}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved in {save_path}")

    # ID_trainset + OE_trainset
    else:
        for epoch in range(epochs):
            for (id_images, id_labels), (oe_images, _) in zip(data_loaders['id_train_loader'], data_loaders['oe_train_loader']):
                id_images, id_labels, oe_images = id_images.to(device), id_labels.to(device), oe_images.to(device)
                optimizer.zero_grad()
                id_outputs, oe_outputs = model(id_images), model(oe_images)
                loss = criterion(id_outputs, id_labels, oe_outputs)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        print("OE based training completed")

        save_path = f'logs/{config["general"]["model"]}/trained_model/{config["train"]["variant"]}_ood_{config["train"]["method"]}_{config["train"]["id_dataset"]}_{config["train"]["oe_dataset"]}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved in {save_path}")

# execute
if __name__ == "__main__":
    run_ood_train()