import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
training method

1. output of the model: logit_vector = f => 클래스 차원의 벡터
2. f_i = ||f||*f_i: 클래스-i 차원의 값
3. decompose the logit: logit_vector = ||f|| * f/||f||
4. f_hat = f/||f||

<code-flow>
1. load the model
2. load the data
3. get the output of the model
4. normalize the output
5. get the score
'''

class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04):
        super(LogitNormLoss, self).__init__()
        self.tau = tau

    def forward(self, logits, labels):
        # get output(logits) of the model
        logits = logits.to(device)  # (batch x channel)
        labels = labels.to(device)  # (batch)

        # normalize the output
        magnitude = torch.norm(logits, p=2, dim=1, keepdim=True)  # (batch x 1)
        logit_norm = logits / (magnitude + 1e-7)  # (batch x channel)

        # logit_norm_loss
        logit_norm_loss = F.cross_entropy(logit_norm / self.tau, labels)
        return logit_norm_loss
