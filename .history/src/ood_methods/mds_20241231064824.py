from .base_ood import BaseOOD
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDS(BaseOOD):
    def __init__(self, model):
        super().__init__(model)
        self.num_classes = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_features = {cls: [] for cls in range(self.num_classes)}
        self.id_cls_means = []
        self.id_covariances = None

    # # hook
    # def hook_function(self):
    #     def hook(_model, _input, output):
    #         self.penultimate_layer = self.avg_pool(output).squeeze()  # (batch x channel)
    #     return hook

    # method
    def get_class_features(self, id_dataloader):
        for images, labels in id_dataloader:
            images, labels = images.to(device), labels.to(device)  # (batch)
            print("input shape: ", images.shape)
            print("labels shape: ", labels.shape)
            self.model(images)
            output = self.penultimate_layer.flatten(1)  # (batch x channel)
            print("output shape: ", output.shape)

            for i, label in enumerate(labels):
                class_index = label.item()
                self.class_features[class_index].append(output[i])  # {output[i] : (channel)}
                # print("check if class_index, output[i] matches!")
        return self.class_features

    def get_cls_means(self, class_features):
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (num_samples_in_each_cls x channel)
            self.id_cls_means.append(torch.mean(class_data, dim=0))  # (channel)

        self.id_cls_means = torch.stack(self.id_cls_means)  # Convert list to tensor
        print("shape of id_cls_means: ", self.id_cls_means.shape)  # (num_class x channel)
        return self.id_cls_means

    def get_covariance(self, class_features):
        '''
        수정사항!!
        cls_deviations = class_data[cls] - self.id_cls_means
        => cls_deviations = class_stacks[cls] - self.id_cls_means[cls]
        '''
        class_stacks = []  # list of class_data: (num_class)
        for cls in range(self.num_classes):
            class_data = torch.stack(class_features[cls], dim=0)  # (num_sample_in_each_cls x channel)
            class_stacks.append(class_data)

        total_stack = torch.cat(class_stacks, dim=0)  # (total_sample, channel)
        N = total_stack.shape[0]
        print("N = ", N)

        # 수정 후
        cls_deviations = []  # (num_class)
        for cls in range(self.num_classes):
            cls_deviation = class_stacks[cls] - self.id_cls_means[cls]  # (sample x channel)
            cls_deviations.append(cls_deviation)
            print(f"shape of cls_deviation {cls}: ", cls_deviation.shape)

        total_deviations = torch.cat(cls_deviations)  # (num_samples_in_each_cls x channel)
        print("shape of total_deviation: ", total_deviations.shape)  # (N x 512)??  cifar10: (10000 x 512)?


        total_einsum = torch.einsum('Ni, Nj -> ij', total_deviations, total_deviations)
        self.id_covariances = total_einsum / N

        # 수정 전!
        # cls_covariances = []
        # for cls in range(self.num_classes):  # calculate covariance of each classes
        #     cls_deviations = class_stacks[cls] - self.id_cls_means[cls]  # (sample x channel)
        #     # cls_deviations: 각 클래스의 샘플 수와 무관하게 (channel x channel)
        #     cls_covariances.append(torch.einsum('si, sj -> ij', cls_deviations, cls_deviations))  # (channel x channel)

        # stacked_covariance = torch.stack(cls_covariances)  # (num_class x channel x channel)
        # # print("shape of stacked_cls_covariances: ", stacked_covariance.shape) : torch.Size([10, 512, 512])
        # self.id_covariances = torch.stack(cls_covariances).sum(dim=0) / N  # (channel x channel)

        return self.id_covariances

    # apply method
    def apply_method(self, id_loader):
        self.get_class_features(id_loader)
        self.get_cls_means(self.class_features)
        self.get_covariance(self.class_features)

    # compute ood score
    def ood_score(self, images):
        images = images.to(device)
        self.model(images)
        output = self.penultimate_layer.flatten(1)  # (batch x channel)

        id_cls_means = self.id_cls_means  # (class x channel)
        id_covariances = self.id_covariances  # (channel x channel)

        # det = torch.det(id_cls_covariances)
        # if abs(det) < 1e-6:
        #     print("determinant too small")


        batch_deviations = output.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x channel)
        inv_covariance = torch.linalg.inv(id_covariances)  # (channel x channel)
        mahalanobis_distances = torch.einsum('bci,ij,bcj->bc', batch_deviations,
                                             inv_covariance, batch_deviations)  # (batch x class)

        confidence_scores = torch.max(-mahalanobis_distances, dim=1)[0]
        return confidence_scores



# def ood_score(self, inputs, magnitude=0.001):
#     self.model(inputs)
#     output = self.penultimate_layer  # (batch x channel)
#     cls_means = torch.stack(self.id_cls_means).cpu()  # (class x channel)
#     cls_covariances = self.id_cls_covariances.cpu()  # (channel x channel)

#     # Mahalanobis distance calculation
#     batch_deviations = output.unsqueeze(1) - cls_means.unsqueeze(0)  # (batch, class, channel)
#     inv_covariance = torch.inverse(cls_covariances)  # (channel, channel)
#     mahalanobis_distances = torch.einsum('bij,jk,bik->bi', batch_deviations, inv_covariance, batch_deviations)

#     # Get the class with minimal distance
#     min_distance, _ = torch.min(mahalanobis_distances, dim=1)

#     # Compute gradient w.r.t inputs
#     inputs.requires_grad_()
#     loss = torch.mean(min_distance)  # Mahalanobis loss
#     loss.backward()

#     # Add noise based on the gradient
#     gradient = torch.sign(inputs.grad.data)  # Get the sign of the gradient
#     perturbed_inputs = inputs - magnitude * gradient  # Add perturbation

#     # Recompute Mahalanobis distance for perturbed inputs
#     self.model(perturbed_inputs)
#     perturbed_output = self.penultimate_layer
#     perturbed_deviations = perturbed_output.unsqueeze(1) - cls_means.unsqueeze(0)
#     perturbed_distances = torch.einsum('bij,jk,bik->bi', perturbed_deviations, inv_covariance, perturbed_deviations)

#     confidence_scores = -torch.min(perturbed_distances, dim=1)[0]
#     return confidence_scores