import torch
import torch.nn as nn
import torch.nn.functional as F

class MakePatchEmbedding(nn.Module):
    def __init__(self, input_channel, patch_size, dim):
        super(MakePatchEmbedding, self).__init__()

        self.input_channels = input_channel
        self.patch_size = patch_size
        self.linear_proj = nn.Linear(patch_size**2 * input_channel, dim)

    def forward(self, x):

        # turn image to patches in a tensor form
        x = x.unfold(2, self.patch_size, self.patch_size) # unfold along height: unfold(dim-H, Psize, stride)
        x = x.unfold(3, self.patch_size, self.patch_size) # unfold along width: unfold(dim-W, Psize, stride)
        # => shape of x: (batch, channel, num height patch, num width patch, height, width)

        # change shape to (batch, num height patch, num width patch, channel, height, width)
        x.permutate(0,2,3,1,4,5)

        # change shape to (batch, patch, a flattened patch vector)
        x = x.flatten(start_dim=3, end_dim=-1).flatten(start_dim=1, end_dim=2)

        # prepend class_token at the beginning of the patch for each image
        class_token = nn.parameter(torch.ones(x.shape[2]))
        patch_embeddings = torch.cat((class_token, x), dim=1)

        # element-wise add positional vectors to patch embeddings at once by tensor
        positional_tensor = nn.parameter(torch.rand(patch_embeddings.shape))
        final_embeddings = positional_tensor + patch_embeddings

        return final_embeddings # return positional+patch embeddings tensor


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self,):
        pass

class ViT(nn.Module):
    def __init__(self, input_channel):
        super(ViT, self).__init__()

    def forward(self, x):
        pass