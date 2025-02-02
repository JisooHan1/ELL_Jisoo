import yaml

# import yaml file
def load_config(config_path="src/utils/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

print(f"Model: {config['general']['model']}")
print(f"Dataset: {config['general']['dataset']}")
print(f"Augment: {config['general']['augment']}")
print(f"Epoch: {config['general']['epoch']}")
print(f"Batch Size: {config['general']['batch_size']}")
print(f"Variant: {config['train']['variant']}")
print(f"ID Dataset: {config['train']['id_dataset']}")
print(f"OE Dataset: {config['train']['oe_dataset']}")
print(f"OOD Dataset: {config['train']['ood_dataset']}")
print(f"OOD Method: {config['train']['method']}")
