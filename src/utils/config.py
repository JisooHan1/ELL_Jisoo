import yaml
import argparse

def load_config(config_path="src/utils/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def parse_args(config):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=str, help="model name")
    temp_args, _ = parser.parse_known_args()

    main_parser = argparse.ArgumentParser(description="OOD Training Configuration", parents=[parser])

    if temp_args.model:
        model_config = config[temp_args.model]

        main_parser.add_argument("--dataset", type=str, default=model_config["dataset"], help="Dataset name")
        main_parser.add_argument("--augment", action="store_true", default=model_config["augment"], help="Data augmentation flag")

        main_parser.add_argument("--epoch", type=int, default=model_config["epoch"], help="Number of epochs")
        main_parser.add_argument("--batch_size", type=int, default=model_config["batch_size"], help="Batch size")

        main_parser.add_argument("--optimizer", type=str, default=model_config["optimizer"], help="Optimizer")
        main_parser.add_argument("--lr", type=float, default=model_config["lr"], help="Learning rate")
        main_parser.add_argument("--weight_decay", type=float, default=model_config["weight_decay"], help="Weight decay")
        main_parser.add_argument("--momentum", type=float, default=model_config["momentum"], help="Momentum")
        main_parser.add_argument("--scheduler", type=str, default=model_config["scheduler"], help="Scheduler")
        main_parser.add_argument("--milestones", type=list, default=model_config["milestones"], help="Milestones")
        main_parser.add_argument("--gamma", type=float, default=model_config["gamma"], help="Gamma")

    else:
        raise ValueError("Model name is required")

    # ood.test, ood.train
    ood_config = config["ood_config"]
    main_parser.add_argument("--variant", type=str, default=ood_config["variant"], help="Model variant")
    main_parser.add_argument("--id_dataset", type=str, default=ood_config["id_dataset"], help="ID dataset")
    main_parser.add_argument("--oe_dataset", type=str, default=ood_config["oe_dataset"], help="OE dataset")
    main_parser.add_argument("--ood_dataset", type=str, default=ood_config["ood_dataset"], help="OOD dataset")
    main_parser.add_argument("--method", type=str, default=ood_config["method"], help="OOD detection method")
    main_parser.add_argument("--csi", action="store_true", default=ood_config["csi"], help="CSI flag")

    return main_parser.parse_args()

def get_configs():
    config = load_config()
    args = parse_args(config)
    args_dict = vars(args)

    model_config = config[args.model]
    model_config.update({k: args_dict[k] for k in model_config})

    ood_config = config["ood_config"]
    ood_config.update({k: args_dict[k] for k in ood_config})

    return model_config, ood_config

if __name__ == "__main__":
    model_config, ood_config = get_configs()
    print(model_config)
    print(ood_config)

