import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detection")

    parser.add_argument("--model", type=str, required=True, help="model")
    parser.add_argument("--path", type=str, required=False, help="path to the trained model")  # For testing
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--id_dataset", type=str, required=True, help="id dataset")
    parser.add_argument("--oe_dataset", type=str, required=False, help="oe dataset")  # For oe based training
    parser.add_argument("--ood_dataset", type=str, required=True, help="ood dataset")
    parser.add_argument("--method", type=str, help="ood method")

    return parser.parse_args()