import torch
import torch.nn as nn
import torch.nn.functional as F

class MakePatchEmbedding(nn.Module):
    def __init__(self, input_channel, patch_size, dim):
        super(MakePatchEmbedding, self).__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.linear_proj = nn.Linear(patch_size**2 * input_channel, dim)

        self.class_token = nn.Parameter(torch.ones(1, 1, dim))
        self.positional_tensor = nn.Parameter(torch.rand(1, (dim // patch_size) ** 2 + 1, dim))


    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # turn image to patches in a tensor form
        x = x.unfold(2, self.patch_size, self.patch_size) # unfold along height: unfold(dim-H, Psize, stride)
        x = x.unfold(3, self.patch_size, self.patch_size) # unfold along width: unfold(dim-W, Psize, stride)
        # => shape of x: (batch size, channel, num height patch, num width patch, height, width)

        # change shape to (batch size, num height patch, num width patch, channel, height, width)
        x = x.permute(0,2,3,1,4,5)

        # change shape to (batch size, patch, flattened patch vector)
        x = x.flatten(start_dim=3, end_dim=-1).flatten(start_dim=1, end_dim=2)

        # change shape to (batch size, patch, dim)
        x = self.linear_proj(x)

        # prepend class_token at the beginning of the patch for each image

        class_token = self.class_token.expand(batch_size, -1, -1)
        patch_embeddings = torch.cat((class_token, x), dim=1)

        # element-wise add positional vectors to patch embeddings at once by tensor
        final_embeddings =  self.positional_tensor[:, :patch_embeddings.size(1), :] + patch_embeddings

        # return positional+patch embeddings tensor
        return final_embeddings # =>(batch_size, num_patches, dim)

class Attention(nn.Module):
    def __init__(self, dim, dim_qkv):
        super(Attention, self).__init__()

        self.dim_qkv = dim_qkv

        # Wq: (dim, 64)
        self.wq_matrix = nn.Parameter(torch.ones(dim, self.dim_qkv))
        # Wk: (dim, 64)
        self.wk_matrix = nn.Parameter(torch.ones(dim, self.dim_qkv))
        # Wv: (dim, 64)
        self.wv_matrix = nn.Parameter(torch.ones(dim, self.dim_qkv))

        self.softmax = nn.Softmax(dim=-1) # sum of each row is 1

    def forward(self, x):

        Wq = torch.matmul(x, self.wq_matrix) # => (batch_size, num_patches, dim_q)
        Wk = torch.matmul(x, self.wk_matrix) # => (batch_size, num_patches, dim_k)
        Wv = torch.matmul(x, self.wv_matrix) # => (batch_size, num_patches, dim_v)

        scores = torch.einsum('bik,bjk->bij', Wq, Wk) # => (batch_size, num_patches, num_patches)
        attention_weights = self.softmax(scores / self.dim_qkv**0.5) # => (batch_size, num_patches, num_patches)

        attention_output = torch.matmul(attention_weights, Wv) # =>(batch_size, num_patches, dim_v)

        return attention_output

class Encoder(nn.Module):
    def __init__(self, dim, num_head):
        super(Encoder, self).__init__()

        self.dim = dim
        dim_qkv = 64
        self.num_head = num_head
        self.attention = Attention(dim, dim_qkv)
        self.linear = nn.Linear(dim_qkv * num_head, dim)

        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):

        # layer norm & MSA & residual connection
        identity_map_1 = x # =>(batch_size, num_patches, dim)
        x = self.layer_norm_1(x)
        final_attention_result = self.attention(x)
        attention_outputs = [final_attention_result]
        for _ in range(self.num_head - 1):
            out = self.attention(x)
            attention_outputs.append(out)
        final_attention_result = torch.cat(attention_outputs, dim=-1)  # =>(batch_size, num_patches, dim_v*num_head)
        final_attention_result = self.linear(final_attention_result) # =>(batch_size, num_patches, dim)
        out = final_attention_result + identity_map_1

        # layer norm & MLP & residual connection
        identity_map_2 = out # =>(batch_size, num_patches, dim)
        out = self.layer_norm_1(out)
        out = self.fc1(out) # =>(batch_size, num_patches, dim * 4)
        out = F.gelu(out)
        out = self.fc2(out) # =>(batch_size, num_patches, dim)
        out += identity_map_2

        return out # =>(batch_size, num_patches, dim)

class ViT(nn.Module):
    def __init__(self, input_channel):
        super(ViT, self).__init__()

        patch_size = 4
        dim = 192
        num_head = int(dim/64)
        self.depth = 8

        self.patch_embeddings = MakePatchEmbedding(input_channel, patch_size, dim)
        self.encoder = Encoder(dim, num_head)

        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embeddings(x)

        for _ in range(self.depth):
            x = self.encoder(x)

        class_token_output = x[:, 0]  # (batch_size,  dim)
        out = self.fc(class_token_output)
        return out