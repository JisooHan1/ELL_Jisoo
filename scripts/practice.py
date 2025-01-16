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