from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import math
from mmcv.cnn import build_norm_layer

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(image_size=256, channels=3, patch_size= 16, dim= 512, depth= 12, num_classes= 1000, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim)
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )
class generator(nn.Module):

    def __init__(self, image_size=256, channels=3, patch_size= 16, dim= 512, depth= 12, num_classes= 1000, expansion_factor = 4, dropout = 0.):
        self.encoder=MLPMixer()
        embed_dim=1024
        self.conv_0 = nn.Conv2d(
            embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(256, 3, kernel_size=1, stride=1)

        _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc_2 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)
    def forward(self,x):
        ### encoder
        x=MLPMixer(x)
        ### deconder
        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        x = self.conv_2(x)
        x = self.syncbn_fc_2(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        x = self.conv_3(x)
        x = self.syncbn_fc_3(x)
        x = F.relu(x, inplace=True)
        x = self.conv_4(x)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

        return nn.Tanh()(x)