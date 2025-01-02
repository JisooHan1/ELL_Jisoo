from .base_ood import BaseOOD
import torch
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

class LogitNorm(BaseOOD):
    def __init__(self, model):
        super().__init__(model)
        self.tau = None

    # def get_normalized_outputs(self, dataloader):
    #     for images, _ in dataloader:
    #         images = images.to(device)
    #         self.model(images)

    #         outputs = []
    #         outputs.append(self.penultimate_layer.flatten(1))

    #     outputs = torch.cat(outputs, dim=0)
    #     self.normalized_outputs = F.normalize(outputs, p=2, dim=1)
    #     return self.normalized_outputs  # (num_samples, 512)

    def get_normalized_outputs(self, batch_images):
        self.model(batch_images)
        outputs = []
        outputs.append(self.penultimate_layer.flatten(1))
        outputs = torch.cat(outputs, dim=0)
        self.normalized_outputs = F.normalize(outputs, p=2, dim=1)
        return self.normalized_outputs  # (num_samples, 512)

    def ood_score(self, batch_images, model=None):
        batch_images = batch_images.to(device)
        self.get_normalized_outputs(batch_images)
        logits = self.model.fc(self.normalized_outputs)  # (batch, 10)

        softmax = F.softmax(logits, dim=1)  # (batch, 10)
        scores = torch.max(softmax, dim=1)[0]  # (batch,)

        return scores  # (num_samples,)