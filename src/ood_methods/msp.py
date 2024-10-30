import torch
import torch.nn.functional as F

def MSP(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        softmax_scores = F.softmax(outputs, dim=1)  # (batch_size, num_class)
        max_score, _ = torch.max(softmax_scores, dim=1)  # output: MSP, index - (batch_size)

    return max_score  # type: tensor
