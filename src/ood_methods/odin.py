import torch
import torch.nn.functional as F

def odin_score(input_data, model, temperature=1000, epsilon=0.001):
    # make a copy of input_data
    input_data = input_data.clone().detach().requires_grad_(True)

    # forward pass with temperature scaling
    outputs =  model(input_data) / temperature # (batch_size, num_classes)

    # get predicted class
    softmax_scores = F.softmax(outputs, dim=1)
    pred_class = torch.argmax(softmax_scores, dim=1)

    # calculate loss and gradient
    loss = F.cross_entropy(outputs, pred_class)
    loss.backward()

    # perturb input data (update input data)
    perturbed_input = input_data - epsilon * torch.sign(input_data.grad)

    # second forward pass with perturbed input
    with torch.no_grad():
        perturbed_output = model(perturbed_input) / temperature
        calibrated_scores = F.softmax(perturbed_output, dim=1) # (batch_size, num_classes)
        max_score = torch.max(calibrated_scores, dim=1)[0] # (batch_size)

    return max_score