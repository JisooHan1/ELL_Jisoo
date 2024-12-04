import torch

# 2D 텐서 (3x4)
tensor = torch.tensor([[1, 2, 3, 4],
                       [0, 1, 2, 1],
                       [2, 0, 1, 2]])
min_values, min_indices = torch.min(tensor, dim=1)

print(tensor.shape)          # torch.Size([3, 4])
print(min_values.shape)      # torch.Size([4])
print(min_indices.shape)     # torch.Size([4])
print(min_values)           # tensor([0, 0, 1, 1])
print(min_indices)          # tensor([1, 2, 2, 1])

# t1 = torch.tensor([[1, 2],
#                    [3, 4]])  # shape: (2, 2)
# t2 = torch.tensor([[5, 6],
#                    [7, 8]])  # shape: (2, 2)

# # dim=0 (첫 번째 차원에 쌓기)
# stacked_0 = torch.stack([t1, t2], dim=0)
# print("dim=0 shape:", stacked_0.shape)  # torch.Size([2, 2, 2])
# print("dim=0:\n", stacked_0)
# """
# tensor([[[1, 2],    # 첫 번째 텐서 (t1)
#          [3, 4]],
#         [[5, 6],    # 두 번째 텐서 (t2)
#          [7, 8]]])
# """

# # dim=1 (두 번째 차원에 쌓기)
# stacked_1 = torch.stack([t1, t2], dim=1)
# print("\ndim=1 shape:", stacked_1.shape)  # torch.Size([2, 2, 2])
# print("dim=1:\n", stacked_1)
# """
# tensor([[[1, 2],    # 첫 번째 행들
#          [5, 6]],
#         [[3, 4],    # 두 번째 행들
#          [7, 8]]])
# """

# # dim=2 (세 번째 차원에 쌓기)
# stacked_2 = torch.stack([t1, t2], dim=2)
# print("\ndim=2 shape:", stacked_2.shape)  # torch.Size([2, 2, 2])
# print("dim=2:\n", stacked_2)
# """
# tensor([[[1, 5],    # 첫 번째 행의 원소들이 쌍으로
#          [2, 6]],
#         [[3, 7],    # 두 번째 행의 원소들이 쌍으로
#          [4, 8]]])
# """

# # 예제 텐서 생성
# error = torch.tensor([1.0, 2.0, 3.0])
# cls_covariances = torch.tensor([
#     [4.0, 1.0, 2.0],
#     [1.0, 5.0, 3.0],
#     [2.0, 3.0, 6.0]
# ])

# # einsum을 사용하여 이차 형식 계산
# # 'i,ij,j -> '는 i와 j 인덱스를 합쳐 결과를 스칼라로 만듭니다.
# result_einsum = torch.einsum('i,ij,j', error, cls_covariances, error)
# print("error^T * cls_covariances * error (einsum):", result_einsum)

# import torch

# # 예제 텐서 생성
# # error는 (n,) 형태의 벡터
# error = torch.tensor([1.0, 2.0, 3.0])

# # cls_covariances는 (n, n) 형태의 행렬
# cls_covariances = torch.tensor([
#     [4.0, 1.0, 2.0],
#     [1.0, 5.0, 3.0],
#     [2.0, 3.0, 6.0]
# ])

# # 첫 번째 행렬 곱: cls_covariances * error
# intermediate = torch.matmul(cls_covariances, error)
# print("cls_covariances * error:\n", intermediate)

# # 두 번째 행렬 곱: error.T * (cls_covariances * error)
# result = torch.matmul(error, intermediate)
# print("\nerror.T * cls_covariances * error:", result)

