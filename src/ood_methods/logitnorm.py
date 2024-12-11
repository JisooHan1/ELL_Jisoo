import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogitNorm:
    def __init__(self, model, temperature=1.0, eps=1e-7):
        self.model = model
        self.temperature = temperature
        self.eps = eps

    def logitnorm_score(self, inputs, model=None):
        """
        LogitNorm 점수를 계산합니다.
        1. 로짓을 정규화 (L2 norm으로 나눔)
        2. 온도 스케일링 적용
        3. 소프트맥스 적용
        """
        inputs = inputs.to(next(self.model.parameters()).device)

        with torch.no_grad():
            # 로짓 계산
            logits = self.model(inputs)

            # 로짓 정규화
            norm = torch.norm(logits, p=2, dim=1, keepdim=True)
            normalized_logits = logits / (norm + self.eps)

            # 온도 스케일링 적용
            scaled_logits = normalized_logits / self.temperature

            # 소프트맥스 적용
            probabilities = F.softmax(scaled_logits, dim=1)

            # 최대 확률값을 신뢰도 점수로 사용
            scores = torch.max(probabilities, dim=1)[0]

        return scores