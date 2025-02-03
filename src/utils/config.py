import yaml
import argparse

def load_config(config_path="src/utils/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def parse_args(defaults):
    parser = argparse.ArgumentParser(description="OOD Training Configuration")

    # main.py
    general = defaults["general"]
    parser.add_argument("--model", type=str, default=general["model"], help="Model name")
    parser.add_argument("--dataset", type=str, default=general["dataset"], help="Dataset name")
    parser.add_argument("--augment", action="store_true" if not general["augment"] else "store_false", help="Data augmentation flag")
    parser.add_argument("--epoch", type=int, default=general["epoch"], help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=general["batch_size"], help="Batch size")

    # ood.test, ood.train
    train = defaults["train"]
    parser.add_argument("--variant", type=str, default=train["variant"], help="Model variant")
    parser.add_argument("--id_dataset", type=str, default=train["id_dataset"], help="ID dataset")
    parser.add_argument("--oe_dataset", type=str, default=train["oe_dataset"], help="OE dataset")
    parser.add_argument("--ood_dataset", type=str, default=train["ood_dataset"], help="OOD dataset")
    parser.add_argument("--method", type=str, default=train["method"], help="OOD detection method")

    return parser.parse_args()

config = load_config()
args = parse_args(config)
args_dict = vars(args)
config["general"].update({k: args_dict[k] for k in config["general"]})
config["train"].update({k: args_dict[k] for k in config["train"]})

# print config
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
