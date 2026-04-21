"""
models.transformer_block
========================

Single pre-norm transformer block (LayerNorm -> MHSA -> residual -> LayerNorm
-> MLP -> residual) used inside both stages of the hierarchical 3D image
encoder described by Figure 1e. FFN hidden dimension is fixed at 1024 per
equation 9 of the manuscript (biomarker encoder shares the same block).

Contents
--------
* ``TransformerBlock`` -- configurable embedding dim, heads, and fixed
  ``mlp_hidden_dim``; GELU activation; residual connections around attention
  and MLP. Optional ``grid`` argument enables 3D relative-position bias in
  the attention block (manuscript eq 12).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Pre-norm transformer block used by the image encoder (Figure 1e)."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 mlp_hidden_dim: int = 1024,
                 dropout: float = 0.1,
                 grid: Optional[Tuple[int, int, int]] = None):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, grid=grid)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
