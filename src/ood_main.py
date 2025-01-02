import argparse
from ood_training import ood_training
from ood_posthoc import ood_posthoc

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

    # training arguments
    # parser.add_argument("--epoch", type=int, required=True, help="epoch")
    # parser.add_argument("--lr", type=float, required=True, help="learning rate")
    # parser.add_argument("--milestones", type=list, help="milestones")

    args = parser.parse_args()

    if args.mode == 'ood_training':
        print("Starting training method...")
        ood_training(args)
    elif args.mode == 'ood_posthoc':
        print("Starting posthoc method...")
        ood_posthoc(args)

if __name__ == "__main__":
    main()
