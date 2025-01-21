# //////////////////////////////////////* get_id_mean_cov() * //////////////////////////////////////
    # # (1. with hign-dim-einsum)
    # def get_id_mean_cov(self, penul_dict):
    #     cls_datas = []  # list of cls_data for each cls
    #     cls_devs = []  # list of cls_dev for each cls

    #     for cls in range(self.num_cls):
    #         cls_data = torch.stack(penul_dict[cls], dim=0)  # (num_samples_in_cls x channel)
    #         cls_mean = torch.mean(cls_data, dim=0)  # (channel)
    #         # print(f"shape of cls_mean of cls {cls}: ", cls_mean.shape)
    #         self.id_train_cls_means.append(cls_mean)  # list of cls_mean for each cls

    #         cls_dev = cls_data - cls_mean.unsqueeze(0)  # (num_samples_in_cls x channel)
    #         # print(f"shape of cls_dev {cls}: ", cls_dev.shape)
    #         cls_devs.append(cls_dev)
    #         cls_datas.append(cls_data)

    #     self.id_train_cls_means = torch.stack(self.id_train_cls_means, dim=0)  # Convert list to tensor

    #     total_stack = torch.cat(cls_datas, dim=0)  # (total_id_trainset_samples x channel)
    #     N = total_stack.shape[0]  # number of total_id_trainset_samples; cifar10 => (50,000)

    #     total_devs = torch.cat(cls_devs, dim=0)  # (total_id_trainset_samples x channel)

    #     total_einsum = torch.einsum("Ni,Nj->ij", total_devs, total_devs)  # (channel x channel)

    #     self.id_train_covariances = total_einsum / N  # (channel x channel)
    #     self.inverse_id_train_cov = torch.linalg.inv(self.id_train_covariances)

    # # (2. without hign-dim-einsum)
    # def get_id_mean_cov(self, class_features):
    #     cls_datas = []  # list of cls_data for each cls
    #     cls_covs = torch.zeros(512, 512)

    #     for cls in range(self.num_classes):
    #         cls_data = torch.stack(class_features[cls], dim=0)  # (num_samples_in_cls x channel)
    #         cls_mean = torch.mean(cls_data, dim=0)  # (channel)
    #         self.id_train_cls_means.append(cls_mean)  # list of cls_mean for each cls

    #         cls_dev = cls_data - cls_mean.unsqueeze(0)  # (num_samples_in_cls x channel)

    #         for i in range(cls_data.shape[0]):
    #             cls_dev = cls_data[i] - cls_mean  # (channel)
    #             cls_cov = torch.einsum('i,j->ij', cls_dev, cls_dev)
    #             cls_covs += cls_cov

    #         cls_datas.append(cls_data)

    #     self.id_train_cls_means = torch.stack(self.id_train_cls_means, dim=0)  # Convert list to tensor

    #     total_stack = torch.cat(cls_datas, dim=0)  # (total_id_trainset_samples x channel)
    #     N = total_stack.shape[0]  # number of total_id_trainset_samples; cifar10 => (50,000)

    #     self.id_train_covariances = cls_covs / N  # (channel x channel)
    #     self.inverse_id_train_cov = torch.linalg.inv(self.id_train_covariances)

    # # (3. average covariance matrices of 10 cls)
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

    # # (4. Using EmpiricalCovariance)
    # def get_id_mean_cov(self, penul_dict):
    #     cls_datas = []
    #     cls_means = []

    #     for cls in range(self.num_cls):
    #         cls_data = torch.stack(penul_dict[cls], dim=0)
    #         cls_datas.append(cls_data)
    #         cls_means.append(cls_data.mean(dim=0))

    #     self.id_train_cls_means = torch.stack(cls_means, dim=0)  # (10 x 512)
    #     total_datas = torch.cat(cls_datas, dim=0)  # (50000 x 512)
    #     total_means = self.id_train_cls_means.unsqueeze(1).repeat(1, total_datas.shape[0]//self.num_cls, 1).reshape(-1, total_datas.shape[1])  # (50000 x 512)

    #     total_devs = total_datas - total_means
    #     emp_cov = EmpiricalCovariance(assume_centered=False).fit(total_devs.cpu().numpy())
    #     self.inverse_id_train_cov = torch.tensor(emp_cov.precision_, device=device).float()

    #     # Debugging
    #     id_train_covariances = torch.tensor(emp_cov.covariance_, device=device).float()
    #     print(id_train_covariances)
    #     print("determinant of covariance: ", torch.det(id_train_covariances))
    #     _, logdet = torch.slogdet(id_train_covariances)
    #     print("Log determinant:", logdet)
    #     eigenvals, _ = torch.linalg.eig(id_train_covariances)
    #     print("max eigenvalue of covariance: ", eigenvals.real.max())
    #     print("min eigenvalue of covariance: ", eigenvals.real.min())
    #     print("is covariance matrix symmetric?: ", torch.allclose(id_train_covariances.T, id_train_covariances, atol=1e-8))
    #     print("max diagonals of covariance:", torch.diag(id_train_covariances).max())
    #     print("min diagonals of covariance:", torch.diag(id_train_covariances).min())
    #     # print("variances: ", torch.diagonal(id_train_covariances))

    # # (5. Using PCA)
    # def get_id_mean_cov(self, penul_dict):
    #     cls_datas = []  # list of cls_data for each cls

    #     for cls in range(self.num_cls):
    #         cls_data = torch.stack(penul_dict[cls], dim=0)  # (num_samples_in_cls x channel)
    #         cls_mean = torch.mean(cls_data, dim=0)  # (channel)
    #         self.id_train_cls_means.append(cls_mean)  # list of cls_mean for each cls
    #         cls_datas.append(cls_data)

    #     total_stack = torch.cat(cls_datas, dim=0)  # (50000 x 512)
    #     # fitted on penultimate layer outputs
    #     # total_stack_pca = torch.tensor(pca.fit_transform(total_stack.cpu().numpy()), device=device)  # (50000 x n_components)
    #     N = total_stack.shape[0]  # cifar10 => (50,000)
    #     D = total_stack.shape[1]  # cifar10 => 512

    #     self.id_train_cls_means = torch.stack(self.id_train_cls_means, dim=0)  # (10 x 512)
    #     total_mean = self.id_train_cls_means.unsqueeze(1).repeat(1, N, 1).reshape(-1, D)  # (50000 x 512)
    #     # total_mean_mean = torch.mean(total_mean, dim=0).mean()  # 0?
    #     # print(total_mean_mean)
    #     # total_mean_pca = torch.tensor(pca.transform(total_mean.cpu().numpy()),device=device)  # (50000 x n_components)

    #     # total_dev = total_stack_pca - total_mean_pca
    #     total_dev = total_stack - total_mean
    #     self.id_train_covariances = torch.einsum('Ni,Nj->ij', total_dev, total_dev) / N  # (512 x 512)

    #     print(self.id_train_covariances)
    #     print("determinant of covariance: ", torch.det(self.id_train_covariances))
    #     _, logdet = torch.slogdet(self.id_train_covariances)
    #     print("Log determinant:", logdet)
    #     eigenvals, _ = torch.linalg.eig(self.id_train_covariances)
    #     print("max eigenvalue of covariance: ", eigenvals.real.max())
    #     print("min eigenvalue of covariance: ", eigenvals.real.min())
    #     print("is covariance matrix symmetric?: ", torch.allclose(self.id_train_covariances.T, self.id_train_covariances, atol=1e-8))
    #     print("max diagonals of covariance:", torch.diag(self.id_train_covariances).max())
    #     print("min diagonals of covariance:", torch.diag(self.id_train_covariances).min())
    #     # print("variances: ", torch.diagonal(self.id_train_covariances))

    #     self.inverse_id_train_cov = torch.linalg.inv(self.id_train_covariances)

# //////////////////////////////////////* ood_score() * //////////////////////////////////////

    # # (1. with input pre-processing)
    # # compute ood score
    # def ood_score(self, images):
    #     id_cls_means = self.id_train_cls_means  # (class x channel)
    #     inv_covariance = self.inverse_id_train_cov
    #     images = images.to(device)

    #     with torch.set_grad_enabled(True):~
    #         images = images.clone().detach().requires_grad_(True)
    #         images.grad = None

    #         # forward pass
    #         self.model(images)
    #         output = self.penultimate_layer  # (batch x channel)

    #         # compute mahalanobis distance
    #         test_devs = output.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x channel)
    #         mahalanobis_distances = torch.einsum('bci,ij,bcj->bc', test_devs, inv_covariance, test_devs)  # (batch x class)

    #         # compute loss
    #         loss = torch.min(mahalanobis_distances, dim=1).values.mean()
    #         loss.backward()
    #         preturbed_images = images - self.epsilon * torch.sign(images.grad)

    #     with torch.no_grad():
    #         self.model(preturbed_images)  # forward pass
    #         output = self.penultimate_layer  # (batch x channel)
    #         test_devs = output.unsqueeze(1) - id_cls_means.unsqueeze(0)  # (batch x class x channel)
    #         mahalanobis_distances = torch.einsum('bci,ij,bcj->bc', test_devs, inv_covariance, test_devs)  # (batch x class)

    #     mds_scores, _ = torch.max(-mahalanobis_distances, dim=1)  # (batch)
    #     return mds_scores

    # # (2. without input pre-processing, without high-dim-einsum)
    # # compute ood score
    # def ood_score(self, images):
    #     id_cls_means = self.id_train_cls_means  # (class x channel)
    #     inv_covariance = self.inverse_id_train_cov
    #     images.to(device)
    #     self.model(images)

    #     output = self.penultimate_layer  # (batch x channel)
    #     final_test_covs = []
    #     for i in range(output.shape[0]):
    #         test_cov=[]
    #         for cls in range(self.num_classes):
    #             test_dev = output[i] - id_cls_means[cls]  # (channel)
    #             test_cov.append(torch.einsum('i,ij,j->', test_dev, inv_covariance, test_dev).unsqueeze(0))  # a value
    #         final_test_covs.append(min(test_cov))

    #     concat_covs = torch.cat(final_test_covs, dim=0)
    #     mds_scores = -concat_covs
    #     return mds_scores