"""
models.attention
================

Scaled multi-head self-attention used across the CardioAI encoders
(biomarker encoder, Figure 1d; and every transformer block inside the
image encoder, Figure 1e). Supports optional 3D relative position bias
``B_spatial`` per manuscript equation 12.

Contents
--------
* ``MultiHeadSelfAttention`` -- QKV + softmax + output projection, with an
  optional learnable relative position bias table indexed by (Dd, Dh, Dw)
  offsets over a 3D voxel grid. When ``grid`` is omitted the block behaves
  as a standard transformer attention.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional 3D relative position bias."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 grid: Optional[Tuple[int, int, int]] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.grid = grid
        if grid is not None:
            # Relative-position-bias table for a 3D token grid (manuscript
            # eq 12). Size: (2Dd - 1) * (2Dh - 1) * (2Dw - 1) entries per
            # head, indexed by the (dd, dh, dw) offset between query and
            # key tokens.
            dd, dh, dw = grid
            table_size = (2 * dd - 1) * (2 * dh - 1) * (2 * dw - 1)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(table_size, num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            # Pre-compute flat index from each (qi, kj) pair to the
            # corresponding bias-table row. Stored as a buffer so it moves
            # with the module but is not a learnable parameter.
            coords_d = torch.arange(dd)
            coords_h = torch.arange(dh)
            coords_w = torch.arange(dw)
            gd, gh, gw = torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
            coords = torch.stack([gd.flatten(), gh.flatten(), gw.flatten()], dim=0)  # (3, N)
            relative = coords[:, :, None] - coords[:, None, :]  # (3, N, N)
            relative[0] += dd - 1
            relative[1] += dh - 1
            relative[2] += dw - 1
            flat = (relative[0] * (2 * dh - 1) * (2 * dw - 1) +
                    relative[1] * (2 * dw - 1) +
                    relative[2])
            self.register_buffer("relative_position_index", flat, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if self.grid is not None:
            # (N, N, num_heads) -> (num_heads, N, N) -> (1, num_heads, N, N)
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)
            attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
