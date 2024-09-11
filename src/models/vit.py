import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()

        self.dim = embed_dim
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size**2 * in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim).mul_(0.02))
        self.positional_tensor = nn.Parameter(torch.rand(1, (img_size // patch_size)**2 + 1, embed_dim).mul_(0.02))


    def forward(self, x):
        device = x.device
        batch_size, channels, height, width = x.size()

        # Number of patches
        num_patches_h =  height // self.patch_size
        num_patches_w =  width // self.patch_size

        # (batch size, channels, height, width)
        # =>(batch size, channels, num_patches_h, height, num_patches_w, width)
        x = x.view(batch_size, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)

        # => (batch size, num_patches_h, num_patches_w, channel, height, width)
        x = x.permute(0,2,4,1,3,5)

        # => (batch size, patches, flattened patch vector)
        x = x.contiguous().view(batch_size, num_patches_h*num_patches_w, -1)

        # => (batch size, patches, dim)
        x = self.projection(x)

        # Prepend class token
        cls_token = self.cls_token.expand(batch_size, -1, -1) # => (batch size, 1, dim)
        patch_embeddings = torch.cat((cls_token, x), dim=1) # => (batch size, patches + 1, dim)

        # element-wise add positional vectors
        final_embeddings =  self.positional_tensor[:, :patch_embeddings.size(1), :] + patch_embeddings

        # return positional+patch embeddings tensor
        return final_embeddings # =>(batch_size, num_patches, dim)

class MSA(nn.Module):
    def __init__(self, heads, embed_dim):
        super(MSA, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads

        # Wq: (embed_dim, embed_dim)
        self.wq_matrix = nn.Parameter(torch.ones(embed_dim, embed_dim).mul_(0.02))
        # Wk: (embed_dim, embed_dim)
        self.wk_matrix = nn.Parameter(torch.ones(embed_dim, embed_dim).mul_(0.02))
        # Wv: (embed_dim, embed_dim)
        self.wv_matrix = nn.Parameter(torch.ones(embed_dim, embed_dim).mul_(0.02))

        self.softmax = nn.Softmax(dim=-1) # sum of each row is 1

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        # => (batch_size, num_patches, heads, embed_dim)
        Wq = torch.matmul(x, self.wq_matrix).view(batch_size, num_patches, self.heads, self.head_dim)
        Wk = torch.matmul(x, self.wk_matrix).view(batch_size, num_patches, self.heads, self.head_dim)
        Wv = torch.matmul(x, self.wv_matrix).view(batch_size, num_patches, self.heads, self.head_dim)

        # => (batch_size, heads, num_patches, embed_dim)
        Wq = Wq.permute(0, 2, 1, 3)
        Wk = Wk.permute(0, 2, 1, 3)
        Wv = Wv.permute(0, 2, 1, 3)

        scores = torch.einsum('bhid,bhjd->bhij', Wq, Wk) # => (batch_size, heads, num_patches, num_patches)
        attention_weights = self.softmax(scores / self.head_dim**0.5) # => (batch_size, heads, num_patches, num_patches)
        msa = torch.einsum('bhij,bhjd->bhid', attention_weights, Wv)  # (batch_size, heads, num_patches, head_dim)

        # => (batch_size, num_patches, embed_dim)
        msa = msa.permute(0, 2, 1, 3).contiguous()
        msa = msa.view(batch_size, num_patches, embed_dim)

        return msa

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_head):
        super(Encoder, self).__init__()

        self.dim = embed_dim
        self.num_head = num_head
        self.attention = MSA(num_head, embed_dim)
        self.linear = nn.Linear(embed_dim * num_head, embed_dim)

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):

        # 1) layer norm & MSA & residual connection
        identity_1 = x # => (batch_size, num_patches, embed_dim)
        x = self.layer_norm_1(x)
        msa = self.attention(x)
        x = msa + identity_1

        # 2) layer norm & MLP & residual connection
        identity_2 = x # =>(batch_size, num_patches, dim)
        x = self.layer_norm_2(x)
        x = self.fc1(x) # =>(batch_size, num_patches, dim * 4)
        x = F.gelu(x)
        x = self.fc2(x) # =>(batch_size, num_patches, dim)
        x += identity_2

        return x # =>(batch_size, num_patches, dim)

class ViT(nn.Module):
    def __init__(self, in_channels, img_size):
        super(ViT, self).__init__()

        patch_size = 4
        dim = 192
        num_head = int(dim/64)
        self.depth = 8

        self.patch_embeddings = PatchEmbedding(in_channels, img_size, patch_size, dim)
        self.encoder = Encoder(dim, num_head)

        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embeddings(x)

        for _ in range(self.depth):
            x = self.encoder(x)

        class_token_output = x[:, 0]  # (batch_size,  dim)
        out = self.fc(class_token_output)
        return out