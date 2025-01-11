import argparse
from ood_posthoc import run_ood_posthoc_method
from ood_training import run_ood_training_method

def main():
    parser = argparse.ArgumentParser(description="OOD Detection")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['ood_training', 'ood_posthoc'] ,help="ood train mode or ood test mode")

    # common arguments
    parser.add_argument("--model", type=str, required=True, help="model")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--id_dataset", type=str, required=True, help="id dataset")
    parser.add_argument("--ood_dataset", type=str, required=True, help="ood dataset")
    parser.add_argument("--method", type=str, help="ood method")
    parser.add_argument("--train", type=str, help="train or not")

    args = parser.parse_args()

    # ood_posthoc
    if args.mode == 'ood_posthoc':
        print("Starting posthoc method...")
        run_ood_posthoc_method(args)

    # ood_training
    elif args.mode == 'ood_training':
        print("Starting training method...")
        run_ood_training_method(args)


if __name__ == "__main__":
    main()
