import torch
import torch.nn.functional as F
from models import ResNet
from datasets import load_dataset
from torcheval.metrics import BinaryAUROC, BinaryAUPRC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReActDetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.activations = {}
        self.register_hooks()
        self.c = None

    def load_model(self, model_path):
        model = ResNet(3).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def get_activation(self, name):
        def hook(_model, _input, output):
            self.activations[name] = output.detach().to(device)
        return hook

    def register_hooks(self):
        self.model.GAP.register_forward_hook(self.get_activation('penultimate'))

    def calculate_threshold(self, dataloader):
        # get penultimate activations for ID data
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                self.model(inputs)

        penultimate = self.activations['penultimate'].flatten(1)
        self.c = torch.quantile(penultimate, 0.9, dim=0).to(device)

    def react(self, dataloader):
        scores_list = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                self.model(x)

                penultimate = self.activations['penultimate'].flatten(1)
                clamped = torch.clamp(penultimate, max=self.c.unsqueeze(0))

                logits = self.model.fc(clamped)
                softmax = F.softmax(logits, dim=1)
                scores = torch.max(softmax, dim=1)[0]
                scores_list.append(scores)
        return torch.cat(scores_list)

    def evaluate_ood_detection(self, id_scores, ood_scores):
        labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)])
        scores = torch.cat([id_scores, ood_scores])

        binary_auroc = BinaryAUROC().to(device)
        binary_auroc.update(scores, labels)
        binary_auprc = BinaryAUPRC().to(device)
        binary_auprc.update(scores, labels)

        auroc = binary_auroc.compute()
        aupr = binary_auprc.compute()
        return auroc, aupr

def main():
    # load datasets
    id_trainset, id_testset = load_dataset("CIFAR10")
    _ood_trainset, ood_testset = load_dataset("SVHN")

    id_train_loader = torch.utils.data.DataLoader(id_trainset, batch_size=64, shuffle=True, num_workers=2)
    id_test_loader = torch.utils.data.DataLoader(id_testset, batch_size=64, shuffle=False, num_workers=2)
    ood_test_loader = torch.utils.data.DataLoader(ood_testset, batch_size=64, shuffle=False, num_workers=2)

    # initialize ReAct detector
    model_path = "logs/ResNet/trained_model/trained_resnet_20241009_185530.pth"
    detector = ReActDetector(model_path)

    # calculate threshold using ID training data
    detector.calculate_threshold(id_train_loader)

    # evaluate OOD detection
    id_scores = detector.react(id_test_loader)
    ood_scores = detector.react(ood_test_loader)

    auroc, aupr = detector.evaluate_ood_detection(id_scores, ood_scores)
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPR: {aupr:.4f}')

if __name__ == "__main__":
    main()