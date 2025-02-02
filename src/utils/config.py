import yaml

# import yaml file
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

print(f"Model: {config['general']['model']}")
print(f"Dataset: {config['general']['dataset']}")
print(f"Epochs: {config['general']['epoch']}")
print(f"Batch Size: {config['general']['batch_size']}")
print(f"OOD Method: {config['train']['method']}")
