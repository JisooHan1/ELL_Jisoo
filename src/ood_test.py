import torch
from datasets.datasets import get_data_loaders
from models import load_saved_model, model_path
from ood_methods import get_ood_methods
from utils.ood_configs import posthoc_methods
from utils.ood_metrics import evaluations
from utils.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

def get_ood_scores(loader, ood_method):
    scores = []
    for images, _ in loader:
        images = images.to(device)
        batch_scores = ood_method.ood_score(images).cpu().numpy()  # (batch)
        scores.append(batch_scores)
    return scores

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

    print("Loading data...")
    data_loaders, _, _ = get_data_loaders(id_dataset, None, ood_dataset, batch_size, augment)
    id_train_loader = data_loaders['id_train_loader']
    id_test_loader = data_loaders['id_test_loader']
    ood_test_loader = data_loaders['ood_test_loader']

    print("Applying posthoc methods...")
    if method in posthoc_methods:
        ood_method = get_ood_methods(method, model)
        ood_method.apply_method(id_train_loader)
    else:
        ood_method = get_ood_methods('msp', model)  # apply msp method by default

    print("Computing scores...")
    id_scores = get_ood_scores(id_test_loader, ood_method)
    ood_scores = get_ood_scores(ood_test_loader, ood_method)

    # get FPR95, AUROC and AUPR
    results = evaluations(id_scores, ood_scores)
    print(results)  # example: {'FPR95': 0.001, 'AUROC': 0.999, 'AUPR': 0.999}

if __name__ == "__main__":
    run_ood_test()