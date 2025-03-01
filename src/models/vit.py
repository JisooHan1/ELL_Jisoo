import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()

        self.dim = embed_dim
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size**2 * in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim).mul_(0.02))
        self.positional_tensor = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim).mul_(0.02))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Number of patches
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # (batch, channel, patches_h, height, patches_w, width)
        x = x.view(batch_size, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)

        # (batch, patches_h, patches_w, channel, height, width)
        x = x.permute(0, 2, 4, 1, 3, 5)

        # (batch, patches, flattened patch vector)
        x = x.contiguous().view(batch_size, num_patches_h * num_patches_w, -1)

        # (batch, patches, dim)
        x = self.projection(x)

        # Prepend class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # => (batch, 1, dim)
        patch_embeddings = torch.cat((cls_token, x), dim=1)  # => (batch, patches + 1, dim)

        # element-wise add positional vectors
        final_embeddings = self.positional_tensor + patch_embeddings  # broadcasting

        return final_embeddings  # (batch, patches + 1, dim)


class MSA(nn.Module):
    def __init__(self, heads, embed_dim):
        super(MSA, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads

        # Wq, Wk, Wv in one tensor
        self.qkv_matrix = nn.Linear(embed_dim, 3 * embed_dim)
        self.softmax = nn.Softmax(dim=-1)  # sum of each col is 1

        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        qkv = self.qkv_matrix(x)  # (batch_size x num_patches x 3*embed_dim)
        qkv = qkv.view(batch_size, num_patches, 3, self.heads, self.head_dim)  # (batch_size x num_patches x 3 x heads x head_dim)

        Wq, Wk, Wv = torch.chunk(qkv, 3, dim=2)  # (batch_size x num_patches x 1 x heads x head_dim) x3
        Wq, Wk, Wv = Wq.squeeze(2), Wk.squeeze(2), Wv.squeeze(2)  # (batch_size x num_patches x heads x head_dim)

        scores = torch.einsum("bihd,bjhd->bhij", Wq, Wk)  # (batch_size, heads, num_patches, num_patches)
        attention_weights = self.softmax(scores / self.head_dim**0.5)
        msa = torch.einsum("bhij,bjhd->bihd", attention_weights, Wv)  # (batch_size, num_patches, heads, head_dim)

        msa = msa.contiguous().view(batch_size, num_patches, embed_dim)  # (batch_size x num_patches x embed_dim)

        return self.linear(msa)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_head):
        super(TransformerBlock, self).__init__()

        self.dim = embed_dim
        self.num_head = num_head
        self.attention = MSA(num_head, embed_dim)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.fc_1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc_2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        # 1) layer norm & MSA & residual connection
        x = x + self.attention(self.ln_1(x))

        # 2) layer norm & MLP & residual connection
        x = x + self.fc_2(F.gelu(self.fc_1(self.ln_2(x))))
        return x  # (batch_size, num_patches, dim)


class ViT(nn.Module):
    def __init__(self, in_channels, img_size):
        super(ViT, self).__init__()

        patch_size = 4
        dim = 192
        num_head = int(dim / 64)  # 3 heads
        self.depth = 8

        self.patch_embeddings = PatchEmbedding(in_channels, img_size, patch_size, dim)
        self.transformer_layer = nn.Sequential(*(TransformerBlock(dim, num_head) for _ in range(self.depth)))

        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = self.ln(self.transformer_layer(x))
        class_token_output = x[:, 0]  # (batch_size, dim)
        return self.fc(class_token_output)
