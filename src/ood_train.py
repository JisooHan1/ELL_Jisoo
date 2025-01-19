import torch
from datasets import load_data
from models import load_model
from utils.ood_configs import get_training_config
from utils.parser import parse_args

# device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# hyperparameters
def initialize_training(args, device):
    data_loaders, id_input_channels, id_image_size = load_data(
        args.id_dataset, args.oe_dataset, args.ood_dataset, args.batch_size, args.augment)

    model = load_model(args.model, id_input_channels, id_image_size).to(device)

    config = get_training_config(args.method)
    criterion = config['criterion']()
    optimizer = config['optimizer'](
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'], nesterov=True)

    # schedulers
    if config['scheduler_type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, milestones=config['milestones'], gamma=config['gamma'])
    elif config['scheduler_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    else:
        raise ValueError(f"Unsupported scheduler type: {config['scheduler_type']}")

    return model, data_loaders, criterion, optimizer, scheduler, config['epochs']

# training
def run_ood_train(args):
    device = get_device()
    model, data_loaders, criterion, optimizer, scheduler, epochs = initialize_training(args, device)

    print("Start training... method: ", args.method)

    # Only ID_trainset
    if args.oe_dataset is None:
        for epoch in range(epochs):
            model.train()
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

        model_path = f'logs/{args.model}/trained_model/ood_{args.method}_{args.id_dataset}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved in {model_path}")

    # ID_trainset + OE_trainset
    else:
        for epoch in range(epochs):
            model.train()
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

        model_path = f'logs/{args.model}/trained_model/ood_{args.method}_{args.id_dataset}_{args.oe_dataset}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved in {model_path}")

# execute
if __name__ == "__main__":
    args = parse_args()
    run_ood_train(args)