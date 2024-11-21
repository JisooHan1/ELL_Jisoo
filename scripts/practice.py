import torch

# penultimate 텐서의 모양: (4, 512) - 4개 샘플, 각각 512차원의 특성 벡터
penultimate = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],    # 실제로는 512개의 값이 있음
    [0.3, 0.4, 0.5, 0.6],    # 예시를 위해 4개만 표시
    [0.5, 0.6, 0.7, 0.8],
    [0.7, 0.8, 0.9, 1.0]
])

# dim=0으로 90번째 백분위수 계산
# dim=0은 수직(열) 방향으로 계산함을 의미
result = torch.quantile(penultimate, 0.9, dim=0)
print(result)
# result의 모양: (512,) - 각 특성별로 90번째 백분위수 값을 가짐