import torch
import numpy as np
from datasets.datasets import get_data_loaders
from models import load_model, load_saved_model, model_path
from utils.ood_configs import get_training_config
from utils.config import get_configs

# device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# initialize training
def initialize_training(device):
    model_config, ood_config = get_configs()
    print(f"Model config: {model_config}")
    print(f"OOD config: {ood_config}")

    # data
    data_loaders, id_input_channels, id_image_size = get_data_loaders(
        id_dataset=ood_config['id_dataset'],
        oe_dataset=ood_config['oe_dataset'],
        ood_dataset=ood_config['ood_dataset'],
        batch_size=model_config['batch_size'],
        augment=model_config['augment'],
        csi=ood_config['csi']
    )
    # model
    if ood_config['variant'] == None:
        model = load_model(model_config['model'], id_input_channels, id_image_size).to(device)
    else:
        model = load_saved_model(
            model_config['model'],
            model_path[model_config['model']][ood_config['variant']],
            device
        )

    # training options
    training_options = get_training_config(ood_config['method'])
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
        scheduler = torch.optim.lr_schesduler.CosineAnnealingLR(
            optimizer,
            T_max=training_options['T_max'],
            eta_min=training_options['eta_min']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {training_options['scheduler_type']}")

    if scheduler is None:
        raise ValueError(f"Unsupported scheduler type: {training_options['scheduler']}")

    results = {
        'model': model,
        'device': device,
        'data_loaders': data_loaders,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'model_config': model_config,
        'ood_config': ood_config
    }

    return results

def train_model(results):
    model = results['model']
    device = results['device']
    data_loaders = results['data_loaders']
    criterion = results['criterion']
    optimizer = results['optimizer']
    scheduler = results['scheduler']
    model_config = results['model_config']
    ood_config = results['ood_config']

    for epoch in range(model_config['epoch']):
        for batch in data_loaders['id_train_loader']:
            optimizer.zero_grad()

            if ood_config['method'] == 'csi':
                inputs, labels = batch
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                batch = torch.cat(inputs, dim=0)
                batch_labels = torch.cat([labels] * 8, dim=0)

                outputs = model(batch)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            elif ood_config['method'] in ['moe', 'oe']:
                (id_images, id_labels), (oe_images, _) = batch
                id_images, id_labels, oe_images = id_images.to(device), id_labels.to(device), oe_images.to(device)

                ratio = np.random.beta(1.0, 1.0)
                mixed_images = ratio * id_images + (1 - ratio) * oe_images

                id_outputs = model(id_images)
                oe_outputs = model(oe_images)
                mixed_outputs = model(mixed_images)

                if ood_config['method'] == 'moe':
                    loss = criterion(id_outputs, id_labels, mixed_outputs, ratio)
                elif ood_config['method'] == 'oe':
                    loss = criterion(id_outputs, id_labels, oe_outputs)

            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}/{model_config['epoch']}, Loss: {loss.item():.4f}")

    print("Training completed")
    save_model(model, ood_config['method'], model_config, ood_config)

# save model
def save_model(model, method, model_config, ood_config):
    variant = ood_config['variant']
    id_dataset = ood_config['id_dataset']
    oe_dataset = ood_config['oe_dataset']
    suffix = f'_{id_dataset}_{oe_dataset}' if method in ['moe', 'oe'] else f'_{id_dataset}'
    save_path = f'logs/{model_config["model"]}/trained_model/{variant}_ood_{method}{suffix}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved in {save_path}")

def run_ood_train():
    device = get_device()
    ood_training_config = initialize_training(device)

    print("Start training... method: ", ood_training_config['ood_config']['method'])
    print("Model: ", ood_training_config['model_config']['model'])
    print("Path: ", model_path[ood_training_config['model_config']['model']][ood_training_config['ood_config']['variant']])

    train_model(ood_training_config)

if __name__ == "__main__":
    run_ood_train()