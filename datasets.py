import torchvision
import torchvision.transforms as transforms

def load_dataset(name):
    transform_pars = [transforms.ToTensor(),
                      transforms.RandomResizedCrop(32, scale=(0.5,1))]

    if name == "CIFAR10":
        input_channels = 3
        transform_pars.extend([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10
        
    elif name == "MNIST":
        input_channels = 1
        transform_pars.extend([
            transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.MNIST
        
    else:
        raise ValueError("Invalid dataset name")
    
    transform = transforms.Compose(transform_pars)
    trainset = dataset(root='./data', train=True, download=True, transform=transform)
    testset = dataset(root='./data', train=False, download=True, transform=transform)
    
    return trainset, testset, input_channels
    
    