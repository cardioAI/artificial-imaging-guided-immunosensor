"""
models.image_encoder
====================

3D image encoder implementing Figure 1e. A stack of 4 DICOM sequences shaped
``(B, 4, D, H, W)`` (canonical MASLD grid: ``(B, 4, 32, 96, 96)``) is
transformed into a 256-d L2-normalised embedding via a Swin-Transformer-
style two-stage hierarchy. Sinusoidal 3D positional encodings and relative
position bias (inside each attention block) encode the anatomical layout
per equations 10-13 of the manuscript.

Figure 1e dataflow
------------------
Images -> Patch partition -> Linear embedding -> Transformer block
       -> Patch merging -> Transformer block -> Embedding

Contents
--------
* ``ImageEncoder`` -- configurable patch size, stage dims and depths. Output
  shape ``(B, embed_dim)``, L2-normalised for contrastive alignment with
  biomarker embeddings.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_modules import PatchEmbed3D, PatchMerging3D
from .transformer_block import TransformerBlock


class ImageEncoder(nn.Module):
    """Hierarchical 3D image encoder (Figure 1e)."""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (32, 96, 96),
                 in_channels: int = 4,
                 patch_size: Tuple[int, int, int] = (8, 16, 16),
                 stage1_dim: int = 128,
                 stage1_depth: int = 2,
                 stage2_depth: int = 2,
                 stage1_heads: int = 4,
                 stage2_heads: int = 8,
                 mlp_hidden_dim: int = 1024,
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        d, h, w = input_shape
        pd, ph, pw = patch_size
        assert d % pd == 0 and h % ph == 0 and w % pw == 0, \
            f"input_shape {input_shape} must be divisible by patch_size {patch_size}"
        grid1 = (d // pd, h // ph, w // pw)
        assert all(g % 2 == 0 for g in grid1), \
            f"stage-1 grid {grid1} must be even in every dimension for patch merging"
        n1 = grid1[0] * grid1[1] * grid1[2]
        grid2 = (grid1[0] // 2, grid1[1] // 2, grid1[2] // 2)
        n2 = grid2[0] * grid2[1] * grid2[2]
        stage2_dim = 2 * stage1_dim

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=stage1_dim,
            patch_size=patch_size,
        )
        # Manuscript eq 11: sinusoidal 3D positional encoding, computed
        # once and registered as a non-learnable buffer so checkpoints
        # don't need to persist it.
        self.register_buffer("pos_embed_stage1",
                             _sinusoidal_positional_encoding(n1, stage1_dim),
                             persistent=False)
        self.dropout1 = nn.Dropout(dropout)

        self.stage1_blocks = nn.ModuleList([
            TransformerBlock(stage1_dim, stage1_heads, mlp_hidden_dim, dropout,
                             grid=grid1)
            for _ in range(stage1_depth)
        ])

        self.patch_merging = PatchMerging3D(stage1_dim)
        self.register_buffer("pos_embed_stage2",
                             _sinusoidal_positional_encoding(n2, stage2_dim),
                             persistent=False)
        self.dropout2 = nn.Dropout(dropout)

        self.stage2_blocks = nn.ModuleList([
            TransformerBlock(stage2_dim, stage2_heads, mlp_hidden_dim, dropout,
                             grid=grid2)
            for _ in range(stage2_depth)
        ])

        self.norm = nn.LayerNorm(stage2_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(stage2_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x, grid = self.patch_embed(images)
        x = x + self.pos_embed_stage1
        x = self.dropout1(x)

        for block in self.stage1_blocks:
            x = block(x)

        x, grid = self.patch_merging(x, grid)
        x = x + self.pos_embed_stage2
        x = self.dropout2(x)

        for block in self.stage2_blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        image_embedding = self.projection_head(x)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        return image_embedding


def _sinusoidal_positional_encoding(num_tokens: int, dim: int) -> torch.Tensor:
    """
    Manuscript eq 11: classic Transformer sinusoidal encoding.

        E_pos[p, 2i]   = sin(p / 10000^(2i / d))
        E_pos[p, 2i+1] = cos(p / 10000^(2i / d))

    Returns a (1, num_tokens, dim) tensor suitable for broadcasting into a
    transformer stack. ``dim`` is the per-token embedding dimension; ``p``
    is a flat token index over the 3D voxel grid. Flat indexing is standard
    practice for 3D vision transformers once the voxel grid is linearised,
    and it matches the tensor layout of ``PatchEmbed3D``.
    """
    pe = torch.zeros(num_tokens, dim)
    position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) *
                         (-torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
