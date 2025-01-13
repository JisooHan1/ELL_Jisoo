import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detection")

    # for main.py/ood_test.py/ood_train.py
    parser.add_argument("--model", type=str, required=True, help="model")
    parser.add_argument("--dataset", type=str, required=False, help="dataset")
    parser.add_argument("--augment", type=str, required=False, help="augment data")  # For no augmentation in trainset
    parser.add_argument("--epoch", type=int, required=False, help="number of epochs")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")

    # for ood_test.py/ood_train.py
    parser.add_argument("--path", type=str, required=False, help="path to the trained model")  # For ood testing
    parser.add_argument("--id_dataset", type=str, required=False, help="id dataset")
    parser.add_argument("--oe_dataset", type=str, required=False, help="oe dataset")  # For ood oe based training
    parser.add_argument("--ood_dataset", type=str, required=False, help="ood dataset")
    parser.add_argument("--method", type=str, required=False, help="ood method")

    return parser.parse_args()