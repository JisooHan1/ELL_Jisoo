from .base_ood import BaseOOD
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pca = PCA(n_components=20)

class MDS(BaseOOD):
    def __init__(self, model, epsilon=0.002):
        super().__init__(model)
        self.num_cls = 10
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.penul_dict = {cls: [] for cls in range(self.num_cls)}
        self.id_train_cls_means = []
        self.id_train_covariances = None
        self.inverse_id_train_cov = None
        self.epsilon = epsilon

# //////////////////////////////////////* get_cls_features() * //////////////////////////////////////
    def get_cls_features(self, id_train_loader):
        for images, labels in id_train_loader:
            images, labels = images.to(device), labels.to(device)
            self.model(images)
            output = self.penultimate_layer  # (batch x channel)
            for i, label in enumerate(labels):
                cls_index = label.item()
                self.penul_dict[cls_index].append(output[i])  # {output[i] : (channel)}
        return self.penul_dict


# //////////////////////////////////////* get_id_cls_mean_cov() * //////////////////////////////////////
    # # (average covariance matrices of 10 cls)
    # def get_id_mean_cov(self, penul_dict):
    #     cls_datas = []  # list of cls_data for each cls
    #     cls_einsums = []  # list of cls_einsums for each cls

    #     for cls in range(self.num_cls):
    #         cls_data = torch.stack(penul_dict[cls], dim=0)  # (num_samples_in_cls x channel)
    #         cls_mean = torch.mean(cls_data, dim=0)  # (channel)
    #         self.id_train_cls_means.append(cls_mean)  # list of cls_mean for each cls

    #         cls_dev = cls_data - cls_mean.unsqueeze(0)  # (num_samples_in_cls x channel)
    #         cls_einsums.append(torch.einsum('ni,nj->ij', cls_dev, cls_dev))  # (channel x channel)
    #         cls_datas.append(cls_data)

    #     self.id_train_cls_means = torch.stack(self.id_train_cls_means, dim=0)  # Convert list to tensor

    #     total_stack = torch.cat(cls_datas, dim=0)  # (total_id_trainset_samples x channel)
    #     N = total_stack.shape[0]  # number of total_id_trainset_samples; cifar10 => (50,000)

    #     self.id_train_covariances = torch.stack(cls_einsums, dim=0).sum(dim=0) / N
    #     self.inverse_id_train_cov = torch.linalg.inv(self.id_train_covariances)

    # (Using PCA)
    def get_id_mean_cov(self, penul_dict):
        cls_datas = []  # list of cls_data for each cls

        for cls in range(self.num_cls):
            cls_data = torch.stack(penul_dict[cls], dim=0)  # (num_samples_in_cls x channel)
            cls_mean = torch.mean(cls_data, dim=0)  # (channel)
            self.id_train_cls_means.append(cls_mean)  # list of cls_mean for each cls
            cls_datas.append(cls_data)

        self.id_train_cls_means = torch.stack(self.id_train_cls_means, dim=0)  # (10 x 512)
        total_mean = self.id_train_cls_means.unsqueeze(1).repeat(1, 5000, 1).reshape(-1, 512)  # (50000 x 512)
        total_mean_pca = torch.tensor(pca.transform(total_mean.cpu().numpy()),device=device)  # (50000 x n_components)

        total_stack = torch.cat(cls_datas, dim=0)  # (50000 x 512)
        total_stack_pca = torch.tensor(pca.fit_transform(total_stack.cpu().numpy()), device=device)  # (50000 x n_components)
        N = total_stack.shape[0]  # cifar10 => (50,000)

        total_dev = total_stack_pca - total_mean_pca
        self.id_train_covariances = torch.einsum('Ni,Nj->ij', total_dev, total_dev) / N

        print(self.id_train_covariances)
        print("determinant of covariance: ", torch.det(self.id_train_covariances))
        _, logdet = torch.slogdet(self.id_train_covariances)
        print("Log determinant:", logdet)
        eigenvals, _ = torch.linalg.eig(self.id_train_covariances)
        print("max eigenvalue of covariance: ", eigenvals.real.max())
        print("min eigenvalue of covariance: ", eigenvals.real.min())
        print("is covariance matrix symmetric?: ", torch.allclose(self.id_train_covariances.T, self.id_train_covariances, atol=1e-8))
        print("max diagonals of covariance:", torch.diag(self.id_train_covariances).max())
        print("min diagonals of covariance:", torch.diag(self.id_train_covariances).min())
        # print("variances: ", torch.diagonal(self.id_train_covariances))

        self.inverse_id_train_cov = torch.linalg.inv(self.id_train_covariances)

    # # (Using EmpiricalCovariance)
    # def get_id_mean_cov(self, penul_dict):
    #     cls_datas = []  # list of cls_data for each cls
    #     cls_means = []  # list of cls_mean for each cls

    #     for cls in range(self.num_cls):
    #         cls_data = torch.stack(penul_dict[cls], dim=0).cpu().numpy()  # Convert to NumPy for sklearn
    #         cls_mean = torch.mean(torch.tensor(cls_data), dim=0).cpu().numpy()  # Calculate mean as NumPy
    #         cls_means.append(cls_mean)

    #         cls_datas.append(cls_data)

    #     self.id_train_cls_means = torch.tensor(cls_means, device=device)  # Convert back to tensor

    #     # Flatten all class data for covariance computation
    #     total_stack = torch.cat([torch.tensor(data, device=device) for data in cls_datas], dim=0).cpu().numpy()
    #     N = total_stack.shape[0]  # number of total_id_trainset_samples

    #     # Compute covariance using EmpiricalCovariance
    #     emp_cov = EmpiricalCovariance().fit(total_stack)
    #     self.id_train_covariances = torch.tensor(emp_cov.covariance_, device=device)  # Convert to tensor
    #     print(self.id_train_covariances)
    #     print("determinant of covariance: ", torch.det(self.id_train_covariances))
    #     _, logdet = torch.slogdet(self.id_train_covariances)
    #     print("Log determinant:", logdet)
    #     eigenvals, _ = torch.linalg.eig(self.id_train_covariances)
    #     print("max eigenvalue of covariance: ", eigenvals.real.max())
    #     print("min eigenvalue of covariance: ", eigenvals.real.min())
    #     print("is covariance matrix symmetric?: ", torch.allclose(self.id_train_covariances.T, self.id_train_covariances, atol=1e-8))
    #     # rank = torch.linalg.matrix_rank(self.id_train_covariances)
    #     # print("rank of covariance matrix: ", rank)
    #     print("max diagonals of covariance:", torch.diag(self.id_train_covariances).max())
    #     print("min diagonals of covariance:", torch.diag(self.id_train_covariances).min())
    #     # print("variances: ", torch.diagonal(self.id_train_covariances))

    #     self.inverse_id_train_cov = torch.tensor(emp_cov.precision_, device=device)  # Get inverse covariance directly


# //////////////////////////////////////* apply_method() * //////////////////////////////////////
    # apply method
    def apply_method(self, id_train_loader):
        self.get_cls_features(id_train_loader)
        self.get_id_mean_cov(self.penul_dict)


# //////////////////////////////////////* ood_score() * //////////////////////////////////////
    # (without input pre-processing)
    # compute ood score
    def ood_score(self, images):
        # id_cls_means = self.id_train_cls_means
        id_cls_means = torch.tensor(pca.transform(self.id_train_cls_means.cpu().numpy()), device=device)  # (class x n_components)
        inv_covariance = self.inverse_id_train_cov
        images = images.to(device)
        self.model(images)  # forward pass

        output = self.penultimate_layer  # (batch x channel)
        output_pca = torch.tensor(pca.transform(output.cpu().numpy()),device=device)  # (batch x n_components)

        test_devs = output_pca.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x n_components)
        mahalanobis_distances = torch.einsum('bci,ij,bcj->bc', test_devs, inv_covariance, test_devs)  # (batch x class)

        mds_scores, _ = torch.max(-mahalanobis_distances, dim=1)  # (batch)
        return mds_scores
