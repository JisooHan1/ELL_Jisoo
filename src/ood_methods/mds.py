import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS:
    def __init__(self, model):
        self.model = model
        self.penultimate_outputs = {}  # {'penultimate': (batch, channel)}
        self.num_classes = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_features = {cls: [] for cls in range(self.num_classes)}
        self.cls_means = []
        self.cls_covariances = None
        self.register_hook()

    def get_activation(self, layer_name, output_dict):
        def hook(_model, _input, output):
            pooled_output = self.avg_pool(output).squeeze()
            output_dict[layer_name] = pooled_output
        return hook

    def register_hook(self):
        self.model.GAP.register_forward_hook(
            self.get_activation('penultimate', self.penultimate_outputs)
        )

    def get_class_features(self, id_dataloader):
        for inputs, labels in id_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.model(inputs)
            output = self.penultimate_outputs['penultimate']  # (batch, channel)

            for i, label in enumerate(labels):
                class_index = label.item()
                self.class_features[class_index].append(output[i])  # output[i] : (channel,)

        return self.class_features

    def get_cls_means(self, class_features):
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample, channel)
            self.cls_means.append(torch.mean(class_data, dim=0))  # (channel)
        return self.cls_means

    def get_cls_covariances(self, class_features):
        class_stacks = []  # [(sample, channel), ...]
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (sample, channel)
            class_stacks.append(class_data)

        total_stack = torch.cat(class_stacks, dim=0)  # (total_sample, channel)
        N = total_stack.shape[0]

        class_covariances = []
        for cls in range(self.num_classes):
            deviations = total_stack - self.cls_means[cls].unsqueeze(0)  # (total_sample, channel)
            class_covariances.append(torch.einsum('ni,nj->ij', deviations, deviations))
        self.cls_covariances = torch.stack(class_covariances).sum(dim=0) / N

        return self.cls_covariances

    def mds_score(self, inputs, model=None):
        inputs = inputs.to(device).clone().detach().requires_grad_(True)

        self.model(inputs)
        output = self.penultimate_outputs['penultimate']

        batch_deviations = output.unsqueeze(1) - torch.stack(self.cls_means).unsqueeze(0)
        mahalanobis_distances = torch.einsum('bij,jk,bik->bi', batch_deviations,
                                             torch.inverse(self.cls_covariances), batch_deviations)
        c_hat = torch.argmin(mahalanobis_distances, dim=1)

        batch_size = mahalanobis_distances.shape[0]
        confidence_scores = mahalanobis_distances[torch.arange(batch_size), c_hat]

        return confidence_scores