"""
models.patch_modules
====================

3D patch partition / linear embedding and 3D patch merging -- the two
token-grid transformations that give the image encoder its Swin-style
hierarchical structure (Figure 1e).

Contents
--------
* ``PatchEmbed3D`` -- non-overlapping 3D convolution with ``kernel==stride==patch_size``
  performing both patch partition and linear embedding, followed by a
  LayerNorm on the per-token channel dimension.
* ``PatchMerging3D`` -- concatenates non-overlapping 2x2x2 neighbour tokens
  along the feature dimension (C -> 8C), applies LayerNorm, then projects
  down to 2C. Exact 3D analogue of Swin Transformer's patch merging.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """3D patch partition + linear embedding (Figure 1e, first stage)."""

    def __init__(self,
                 in_channels: int = 4,
                 embed_dim: int = 128,
                 patch_size: Tuple[int, int, int] = (8, 24, 24)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = self.proj(x)
        _, _, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (d, h, w)


class PatchMerging3D(nn.Module):
    """3D patch merging (Figure 1e, middle stage): (B, D*H*W, C) -> (B, D/2*H/2*W/2, 2C)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(8 * in_dim)
        self.reduction = nn.Linear(8 * in_dim, 2 * in_dim, bias=False)

    def forward(self,
                tokens: torch.Tensor,
                grid: Tuple[int, int, int]
                ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        b, n, c = tokens.shape
        d, h, w = grid
        assert n == d * h * w, f"token count {n} does not match grid {grid}"
        assert d % 2 == 0 and h % 2 == 0 and w % 2 == 0, \
            f"grid {grid} must be even in each dimension for 2x2x2 merging"

        x = tokens.view(b, d, h, w, c)
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)

        new_grid = (d // 2, h // 2, w // 2)
        x = x.view(b, -1, 8 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x, new_grid
