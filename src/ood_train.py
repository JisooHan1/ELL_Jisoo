import torch
from datasets import load_data
from models import load_model, save_model
from ood_utils.ood_configs import get_training_config
from ood_utils.ood_parser import parse_args

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class OODTraining:
    def __init__(self, args):
        self.model = args.model                  # ResNet, DenseNet, ...
        self.id_dataset = args.id_dataset        # CIFAR10, SVHN, CIFAR100, ...
        self.oe_dataset = args.oe_dataset        # TinyImageNet, ...
        self.ood_dataset = args.ood_dataset      # CIFAR10, SVHN, CIFAR100, ...
        self.batch_size = args.batch_size        # batch size
        self.method = args.method                # logitnorm, oe, moe ...

    # ood training
    def run_ood_train(self):

        # load data
        data_loaders, id_input_channels, id_image_size = load_data(self.id_dataset, self.oe_dataset, self.ood_dataset, self.batch_size)

        # load model
        model = load_model(self.model, id_input_channels, id_image_size).to(device)  # ResNet, DenseNet

        # method: logitnorm, oe, moe ...
        training_config = get_training_config(self.method)
        print(training_config)

        criterion = training_config['criterion']()  # LogitNormLoss(), OutlierExposureLoss(), MOELoss()
        lr = training_config['lr']
        weight_decay = training_config['weight_decay']
        momentum = training_config['momentum']
        epochs = training_config['epochs']
        optimizer = training_config['optimizer'](model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = training_config['scheduler'](optimizer, milestones=training_config['milestones'], gamma=training_config['gamma'])

        print("Start training... method: ", self.method)

        # Train - Only ID_trainset
        if self.oe_dataset == None:
            for epoch in range(epochs):
                model.train()
                for images, labels in data_loaders['id_train_loader']:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()  # initialize the gradients to '0'
                    outputs = model(images)  # Forward pass => softmax not applied
                    loss = criterion(outputs, labels)  # average loss "over the batch"
                    loss.backward()  # back propagate
                    optimizer.step()  # update weights
                scheduler.step()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            print("Training completed")
            save_model(model, f'logs/{self.model}/trained_model/ood_{self.method}_{self.id_dataset}.pth')

        # Train - ID_trainset + OE_trainset
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
            save_model(model, f'logs/{self.model}/trained_model/ood_{self.method}_{self.id_dataset}_{self.oe_dataset}.pth')

# execute
if __name__ == "__main__":
    args = parse_args()
    ood_training = OODTraining(args)
    ood_training.run_ood_train()