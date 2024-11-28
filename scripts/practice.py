import torch

# 예제 텐서 생성
error = torch.tensor([1.0, 2.0, 3.0])
cls_covariances = torch.tensor([
    [4.0, 1.0, 2.0],
    [1.0, 5.0, 3.0],
    [2.0, 3.0, 6.0]
])

# einsum을 사용하여 이차 형식 계산
# 'i,ij,j -> '는 i와 j 인덱스를 합쳐 결과를 스칼라로 만듭니다.
result_einsum = torch.einsum('i,ij,j', error, cls_covariances, error)
print("error^T * cls_covariances * error (einsum):", result_einsum)

import torch

# 예제 텐서 생성
# error는 (n,) 형태의 벡터
error = torch.tensor([1.0, 2.0, 3.0])

# cls_covariances는 (n, n) 형태의 행렬
cls_covariances = torch.tensor([
    [4.0, 1.0, 2.0],
    [1.0, 5.0, 3.0],
    [2.0, 3.0, 6.0]
])

# 첫 번째 행렬 곱: cls_covariances * error
intermediate = torch.matmul(cls_covariances, error)
print("cls_covariances * error:\n", intermediate)

# 두 번째 행렬 곱: error.T * (cls_covariances * error)
result = torch.matmul(error, intermediate)
print("\nerror.T * cls_covariances * error:", result)

