from .base_ood import BaseOOD
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS(BaseOOD):
    def __init__(self, model, num_classes=10):
        super().__init__(model)
        self.num_classes = num_classes
        self.all_features = []
        self.all_labels = []
        self.id_cls_means = []
        self.id_covariances = []
        self.penultimate_layer = None  # Hook에서 설정될 변수

    # Hook 함수: penultimate_layer에 특징 저장
    def hook_function(self):
        def hook(_model, _input, output):
            self.penultimate_layer = output.flatten(1)  # (batch x channel)
        return hook

    # 특징 추출
    def get_class_features(self, id_dataloader):
        self.model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in id_dataloader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                _ = self.model(images)

                # penultimate_layer에서 특징 추출
                if self.penultimate_layer is None:
                    raise ValueError("Hook is not properly registered. Penultimate layer is None.")

                output = self.penultimate_layer  # (batch x channel)
                all_features.append(output.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        self.all_features = np.concatenate(all_features, axis=0)
        self.all_labels = np.concatenate(all_labels, axis=0)
        return self.all_features, self.all_labels

    # 클래스별 평균 및 공분산 계산
    def compute_gaussian_params(self, all_features, all_labels):
        means = []
        covariances = []

        for cls in range(self.num_classes):
            cls_features = all_features[all_labels == cls]
            if cls_features.shape[0] == 0:
                raise ValueError(f"No features found for class {cls}. Check your dataset or labels.")

            means.append(np.mean(cls_features, axis=0))
            covariances.append(np.cov(cls_features, rowvar=False))

        self.id_cls_means = torch.tensor(np.array(means), dtype=torch.float32, device=device)
        self.id_covariances = torch.tensor(np.array(covariances), dtype=torch.float32, device=device)
        return self.id_cls_means, self.id_covariances

    # 데이터 로더를 사용하여 매서드 적용
    def apply_method(self, id_loader):
        self.get_class_features(id_loader)
        self.compute_gaussian_params(self.all_features, self.all_labels)

    # OOD 점수 계산
    def ood_score(self, images):
        self.model.eval()
        images = images.to(device)

        # Forward pass
        _ = self.model(images)
        if self.penultimate_layer is None:
            raise ValueError("Hook is not properly registered. Penultimate layer is None.")

        output = self.penultimate_layer  # (batch x channel)

        id_cls_means = self.id_cls_means  # (class x channel)
        id_covariances = self.id_covariances  # (class x channel x channel)

        # Mahalanobis 거리 계산
        batch_deviations = output.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x channel)
        inv_covariances = torch.linalg.inv(id_covariances)  # (class x channel x channel)
        mahalanobis_distances = torch.einsum('bci,cij,bcj->bc', batch_deviations, inv_covariances, batch_deviations)  # (batch x class)

        confidence_scores = torch.max(-mahalanobis_distances, dim=1)[0]  # (batch)
        return confidence_scores

    # Hook 등록
    def register_hook(self, layer_name="avgpool"):
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook_function())
                break
        else:
            raise ValueError(f"Layer {layer_name} not found in the model.")
