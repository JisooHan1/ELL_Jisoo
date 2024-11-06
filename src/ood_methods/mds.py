import torch
import torch.nn.functional as F
# from datasets import load_dataset
from models import DenseNet
# import argparse
# from torcheval.metrics import BinaryAUROC, BinaryAUPRC
# from ood_methods import get_ood_methods.

# Load model
def load_model(model_path):
    model = DenseNet(3)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

model = load_model("logs/DenseNet/trained_model/trained_resnet_20241031_004039.pth")


# 모델의 특정 레이어 출력을 저장할 딕셔너리
layer_outputs = {}

# hook 함수 정의
def get_activation(name):
    def hook(model, input, output):
        print(f"Layer: {name}")
        print(f"Input: {input}")  # 튜플 형태로 입력값 출력
        print(f"Output: {output}")  # 출력값 출력
    return hook

# `dense_layers`의 0번째와 1번째 `DensBlock`에 hook 설정
model.dense_layers[0].register_forward_hook(get_activation("dense_layer_0"))
model.dense_layers[1].register_forward_hook(get_activation("dense_layer_1"))

# 임의의 입력 데이터
input_data = torch.randn(1, 3, 32, 32)  # 예를 들어 CIFAR-10 이미지 크기

# 모델에 입력 데이터를 통과시켜 forward hook 실행
_ = model(input_data)

# 각 DensBlock 출력 확인
for layer_name, output in layer_outputs.items():
    print(f"{layer_name}: {output.shape}")


# # Feature extraction
# # Dictionary to store the outputs
# layer_outputs = {}

# # Hook function to capture outputs
# def get_features(name):
#     def hook(model, input, output):
#         layer_outputs[name] = output.detach()
#     return hook

# # Assume `model` is your DenseNet model
# model = DenseNet(3)

# # Register hooks on layers you want to capture
# model.layer1.register_forward_hook(get_features("layer1"))
# model.layer2.register_forward_hook(get_features("layer2"))
# # Repeat for additional layers as needed

# # Forward pass through the model
# input_data = torch.randn(1, 3, 224, 224)  # Example input
# _ = model(input_data)

# # Access captured features
# for layer, output in layer_outputs.items():
#     print(f"{layer} output shape: {output.shape}")

# for each feature,




# def evaluate_ood_detection(id_scores, ood_scores):
#     # generate list of label: ID = 1, OOD = 0
#     labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)])
#     scores = torch.cat([id_scores, ood_scores])

#     # Use Binary metrics for OOD detection
#     binary_auroc = BinaryAUROC()
#     binary_auroc.update(scores, labels)
#     binary_auprc = BinaryAUPRC()
#     binary_auprc.update(scores, labels)

#     auroc = binary_auroc.compute()
#     aupr = binary_auprc.compute()

#     print(f'AUROC: {auroc:.4f}')
#     print(f'AUPR: {aupr:.4f}')

#     def mds_score(data, model, cls_means, covar):
#         global features
#         model.eval()
#         with torch.no_grad():
#             _ = model(data)

#         distances = []
#         for mean in cls_means:
#             diff = features - mean
#             inverse_covar = torch.inverse(covar)
#             distance = torch.sqrt(torch.mm(torch.mm(diff.)))

#     def calculate_stats(model, dataloader, device):
#         global features
#         model.eval()
#         features_list = []
#         labels_list = []

#         with torch.no_grad():
#             for x, y in dataloader:
#                 _ = model(x.to(device))
#                 features_list.append(features)
#                 labels_list.append(y)

#         features = torch.cat(features_list)
#         labels = torch.cat(labels_list)

#         # 클래스별 평균 계산
#         num_classes = len(labels.unique())
#         class_means = []
#         for c in range(num_classes):
#             class_features = features[labels == c]
#             class_mean = class_features.mean(dim=0)
#             class_means.append(class_mean)

#         # 공통 공분산 계산
#         cov_matrix = torch.cov(features.T)
#         return class_means, cov_matrix




# def run_ood_detection(args):
#     # load model
#     model = load_model(args.model_path)
#     batch_size = args.batch_size
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # load ID, OOD data
#     _, id_testset = load_dataset(args.id_dataset)
#     _, ood_testset = load_dataset(args.ood_dataset)

#     # DataLoader - test data
#     id_loader = torch.utils.data.DataLoader(id_testset, batch_size=batch_size, shuffle=True)
#     ood_loader = torch.utils.data.DataLoader(ood_testset, batch_size=batch_size, shuffle=True)

#     # get softmax
#     id_scores = torch.tensor([], device=device)
#     ood_scores = torch.tensor([], device=device)

#     ood_method = get_ood_methods(args.method)

#     for data in id_loader:
#         scores = ood_method(data[0].to(device), model)
#         id_scores = torch.cat([id_scores, scores])

#     for data in ood_loader:
#         scores = ood_method(data[0].to(device), model)
#         ood_scores = torch.cat([ood_scores, scores])

#     # get AUROC and AUPR
#     evaluate_ood_detection(id_scores, ood_scores)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="OOD detection")
#     parser.add_argument("-md", "--model_path", type=str, required=True, help="Path to the trained model file")
#     parser.add_argument("-bs", "--batch_size", type=int, required=True, help="Batch size")
#     parser.add_argument("-id", "--id_dataset", type=str, required=True, help="ID dataset CIFAR10")
#     parser.add_argument("-ood", "--ood_dataset", type=str, required=True, help="OOD dataset SVHN")
#     parser.add_argument("-m", "--method", type=str, help="OOD method to use")
#     args = parser.parse_args()

#     run_ood_detection(args)
