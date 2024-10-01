import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
        def __init__(self, in_channels, patch_size, C):
            super(PatchEmbedding, self).__init__()

            self.patch_size = patch_size
            self.projection = nn.Linear(patch_size**2 * in_channels, C)

        def forward(self, x):
            batch_size, channels, height, width = x.size()

            # Number of patches
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size

            # (batch size, channels, num_patches_h, height, num_patches_w, width)
            x = x.view(batch_size, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)

            # (batch size, num_patches_h, num_patches_w, channels, height, width)
            x = x.permute(0, 2, 4, 1, 3, 5)

            # (batch size, num_patches, channels * height * width)
            x = x.contiguous().view(batch_size, num_patches_h * num_patches_w, -1)

            # (batch size, num_patches(S), dim(C))
            return self.projection(x)


class MixBlock(nn.Module):
        def __init__(self, S, dim_s, C, dim_c):
            super(MixBlock, self).__init__()

            self.ln_1 = nn.LayerNorm(C)
            self.ln_2 = nn.LayerNorm(C)
            self.token_mix = nn.Sequential(
                nn.Linear(S, dim_s),
                nn.GELU(),
                nn.Linear(dim_s, S)
            )
            self.channel_mix = nn.Sequential(
                nn.Linear(C, dim_c),
                nn.GELU(),
                nn.Linear(dim_c, C)
            )

        def forward(self, x):
            x = x + self.token_mix(self.ln_1(x).transpose(1,2)).transpose(1,2)  # token-mix
            x = x + self.channel_mix(self.ln_2(x))  # channel-mix
            return x  # (batch size, num_patches(S), dim(C))

class MLPMixer(nn.Module):
        def __init__(self, in_channels, img_size):
            super(MLPMixer, self).__init__()

            patch_size = 4
            N = 8  # small network size
            S = img_size**2 // patch_size**2  # 64
            C = 256
            dim_s = 128
            dim_c = 512

            self.patch_embeddings = PatchEmbedding(in_channels, patch_size, C)
            self.mix_layer = nn.Sequential(*(MixBlock(S, dim_s, C, dim_c) for _ in range(N)))

            self.fc = nn.Linear(C, 10)

        def forward(self, x):

            x = self.patch_embeddings(x)
            x = self.mix_layer(x)

            # globla average pooling & fc
            x = torch.mean(x, dim=1)  # (batch size, S, C) -> (batch-size, C)
            x = self.fc(x)
            return x
