import torch
import torch.nn.functional as F

def odin_score(input_data, model, temperature=1000, epsilon=0.001):
    input_data.requires_grad = True

    # get y_hat
    outputs_1 =  model(input_data) / temperature
    softmax_scores = F.softmax(outputs_1, dim=1)
    _, pred_class = torch.max(softmax_scores, dim=1)

    # calculate negative_log_softmax socre
    log_softmax_scores = F.log_softmax(outputs_1, dim=1)
    negative_log_softmax = -log_softmax_scores[range(len(pred_class)), pred_class]

    negative_log_softmax.backward(torch.ones_like(negative_log_softmax))

    gradient = input_data.grad
    processed_input_data = input_data - epsilon * torch.sign(gradient)

    with torch.no_grad():
        outputs_2 = model(processed_input_data) / temperature
        calibrated_softmax_scores = F.softmax(outputs_2, dim=1)
        max_score_2, _ = torch.max(calibrated_softmax_scores, dim=1)

    return max_score_2