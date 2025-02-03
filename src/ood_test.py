import torch
from datasets import load_data
from models import load_saved_model, model_path
from ood_methods import get_ood_methods
from utils.ood_configs import posthoc_methods
from utils.ood_metrics import evaluations
from utils.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@torch.no_grad()
def run_ood_test():

    model_name = config['general']['model']
    batch_size = config['general']['batch_size']
    augment = config['general']['augment']

    model_variant = model_path[model_name][config['train']['variant']]
    id_dataset = config['train']['id_dataset']
    ood_dataset = config['train']['ood_dataset']
    method = config['train']['method']

    model = load_saved_model(model_name, model_variant, device)
    model.to(device)

    data_loaders, _, _ = load_data(id_dataset, None, ood_dataset, batch_size, augment)
    id_train_loader = data_loaders['id_train_loader']
    id_test_loader = data_loaders['id_test_loader']
    ood_test_loader = data_loaders['ood_test_loader']

    # apply posthoc methods using id_trainset
    if method in posthoc_methods:
        ood_method = get_ood_methods(method, model)
        ood_method.apply_method(id_train_loader)
    else:
        ood_method = get_ood_methods('msp', model)  # apply msp method by default

    # compute scores
    def compute_scores(loader):
        scores = []
        for images, _ in loader:
            images = images.to(device)
            batch_scores = ood_method.ood_score(images).cpu().numpy()  # (batch)
            scores.append(batch_scores)
        return scores

    id_scores = compute_scores(id_test_loader)
    ood_scores = compute_scores(ood_test_loader)

    # get FPR95, AUROC and AUPR
    results = evaluations(id_scores, ood_scores)
    print(results)  # example: {'FPR95': 0.001, 'AUROC': 0.999, 'AUPR': 0.999}

if __name__ == "__main__":
    run_ood_test()