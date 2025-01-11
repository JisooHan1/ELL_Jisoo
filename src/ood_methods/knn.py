from .base_ood import BaseOOD
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KNN(BaseOOD):
    def __init__(self, model, k=50):
        super().__init__(model)
        self.k = k  # number of neighbors

    # get normalized id_trainset features
    def get_features(self, id_train_loader):
        id_train_features = []
        for images, _ in id_train_loader:
            images = images.to(device)
            self.model(images)
            id_train_features.append(self.penultimate_layer)
        id_train_features = torch.cat(id_train_features, dim=0)  # (id_train_samples x channel)
        self.id_train_features = F.normalize(id_train_features, p=2, dim=1)  # (id_train_samples x channel)
        return self.id_train_features

    # apply method (pre-processing)
    def apply_method(self, id_train_loader):
        self.get_features(id_train_loader)

    # compute ood score
    def ood_score(self, images):  # id_testset/ood_testset
        images = images.to(device)
        self.model(images)

        features = self.penultimate_layer  # (batch x channel)
        l2_features = F.normalize(features, p=2, dim=1)

        distances = torch.cdist(l2_features, self.id_train_features)  # (batch x id_train_samples)
        distances, _ = torch.sort(distances, dim=1, descending=False)

        kth_distance = distances[:, self.k - 1]

        return -kth_distance  # (batch)