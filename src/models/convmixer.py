import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
        def __init__(self, in_channels, P, h):
            super(PatchEmbedding, self).__init__()

            self.p = P
            self.embedding = nn.Conv2d(in_channels=in_channels, out_channels=h, kernel_size=P, stride=P)
            self.gelu = nn.GELU()
            self.bn = nn.BatchNorm2d(h)

        def forward(self, x):
            embeddings = self.embedding(x)  # (batch_size, h, num_patches, num_patches)
            return self.bn(self.gelu(embeddings))


class MixBlock(nn.Module):
        def __init__(self, h):
            super(MixBlock, self).__init__()

            self.depth_wise_conv = nn.Sequential(
                nn.Conv2d(in_channels=h, out_channels=h, kernel_size=5, groups=h, padding=2),
                nn.GELU(),
                nn.BatchNorm2d(h)
            )
            self.point_wise_conv = nn.Sequential(
                nn.Conv2d(in_channels=h, out_channels=h, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(h)
            )

        def forward(self, x):
            x = x + self.depth_wise_conv(x)
            x = self.point_wise_conv(x)
            return x  # (batch_size, h, num_patches, num_patches)

class ConvMixer(nn.Module):
        def __init__(self, in_channels):
            super(ConvMixer, self).__init__()

            P = 2  # patch size
            h = 256  # output channels (dim)
            N = 8  # num layer

            self.patch_embeddings = PatchEmbedding(in_channels, P, h)
            self.mix_layer = nn.Sequential(*(MixBlock(h) for _ in range(N)))

            self.GAP = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(h, 10)

        def forward(self, x):
            x = self.patch_embeddings(x)
            x = self.mix_layer(x)

            # global average pooling & fc
            x = self.GAP(x)  # (batch_size, h, num_patches, num_patches) -> (batch-size, h, 1, 1)
            x = torch.flatten(x, 1)  # (batch-size, h, 1, 1) -> (batch-size, h)
            x = self.fc(x)
            return x
