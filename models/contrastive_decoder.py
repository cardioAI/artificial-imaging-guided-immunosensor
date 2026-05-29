
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

class ContrastiveDecoder(nn.Module):

    def __init__(self,
                 embed_dim: int = 256,
                 out_channels: int = 4,
                 out_shape: Tuple[int, int, int] = (32, 96, 96),
                 seed_shape: Tuple[int, int, int] = (4, 6, 6),
                 base_channels: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.seed_shape = seed_shape
        self.base_channels = base_channels

        sd, sh, sw = seed_shape
        self.seed_linear = nn.Sequential(
            nn.Linear(embed_dim, base_channels * sd * sh * sw),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels, base_channels, kernel_size=(4, 8, 8),
                               stride=(4, 8, 8)),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, biomarker_embedding: torch.Tensor) -> torch.Tensor:
        assert biomarker_embedding.dim() == 2 and biomarker_embedding.size(1) == self.embed_dim, \
            f"expected (B, {self.embed_dim}) got {tuple(biomarker_embedding.shape)}"
        B = biomarker_embedding.size(0)
        sd, sh, sw = self.seed_shape
        seed = self.seed_linear(biomarker_embedding).view(B, self.base_channels, sd, sh, sw)
        x = self.upsample1(seed)
        x = self.upsample2(x)
        x = self.refine(x)

        if tuple(x.shape[-3:]) != self.out_shape:
            x = torch.nn.functional.interpolate(x, size=self.out_shape,
                                                mode="trilinear",
                                                align_corners=False)
        return x
