import torch
from datasets import load_data
from models import load_saved_model, model_path
from ood_methods import get_ood_methods
from ood_utils.ood_configs import posthoc_methods
from ood_utils.ood_metrics import evaluations
from ood_utils.ood_parser import parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OOD_test:
    def __init__(self, args):
        self.model = args.model                            # ResNet, DenseNet
        self.path = model_path[args.model][args.path]      # path to the trained model
        self.batch_size = args.batch_size                  # 16, 32, 64
        self.id_dataset = args.id_dataset                  # CIFAR10, STL10, CIFAR100, SVHN, LSUN, TinyImageNet
        self.ood_dataset = args.ood_dataset                # CIFAR10, STL10, CIFAR100, SVHN, LSUN, TinyImageNet
        self.method = args.method                          # msp, odin, mds, react, logitnorm, knn

    @torch.no_grad()
    def run_ood_test(self):

        # load model
        model = load_saved_model(self.model, self.path, device)

        # load data: id_trainset, id_testset, ood_testset
        data_loaders, _, _ = load_data(self.id_dataset, None, self.ood_dataset, self.batch_size)
        id_train_loader = data_loaders['id_train_loader']
        id_test_loader = data_loaders['id_test_loader']
        ood_test_loader = data_loaders['ood_test_loader']

        # apply posthoc methods using id_trainset
        if self.method in posthoc_methods:
            ood_method = get_ood_methods(self.method, model)
            ood_method.apply_method(id_train_loader)

        else:
            ood_method = get_ood_methods('msp', model)  # apply msp method by default

        # get id_scores and ood_scores
        id_scores = []  # id_testset's scores
        ood_scores = []  # ood_testset's scores

        for images, _ in id_test_loader:
            images = images.to(device)
            batch_id_scores = ood_method.ood_score(images).cpu().numpy()  # (batch)
            id_scores.append(batch_id_scores)

        for images, _ in ood_test_loader:
            images = images.to(device)
            batch_ood_scores = ood_method.ood_score(images).cpu().numpy()  # (batch)
            ood_scores.append(batch_ood_scores)

        # get FPR95, AUROC and AUPR
        results = evaluations(id_scores, ood_scores)
        print(results)  # example: {'FPR95': 0.001, 'AUROC': 0.999, 'AUPR': 0.999}


if __name__ == "__main__":
    args = parse_args()
    ood_test = OOD_test(args)
    ood_test.run_ood_test()

