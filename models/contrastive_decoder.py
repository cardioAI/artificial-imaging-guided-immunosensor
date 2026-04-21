"""
models.contrastive_decoder
==========================

Trained contrastive decoder network that inverts the biomarker encoder
back to MRI voxel space (manuscript: "Artificial images were retrieved
by projecting biomarker embeddings into image space via a trained
contrastive decoder network."). The decoder is a small MLP-to-3D-conv
generator that maps a ``(B, embed_dim)`` biomarker embedding to a
low-resolution 4-channel volume sharing the canonical MASLD grid
``(4, 32, 96, 96)``.

The decoder is trained jointly with the contrastive model via the
retrieval MSE objective (equation 15) -- the loss is applied between the
decoder's reconstruction and the real MRI volume. At inference time it
provides a drop-in replacement for the similarity-weighted retrieval
path used to populate artificial MRI figures.

Contents
--------
* ``ContrastiveDecoder`` -- ``nn.Module`` with a ``forward(biomarker_emb)``
  that returns ``(B, 4, 32, 96, 96)`` volumes.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ContrastiveDecoder(nn.Module):
    """Biomarker embedding -> low-resolution MRI volume."""

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

        # Two ConvTranspose3d stages: (B, base, 4, 6, 6) -> (4, 16, 48, 48)
        # -> (4, 32, 96, 96). Kernel/stride pairs are chosen so the final
        # spatial grid exactly matches ``out_shape`` without extra padding.
        # Stage 1 upsample: factor (2, 8/3, 8/3) -- use stride 2 on D and
        # stride (16/6) = non-integer on H/W, so we compose two stages.
        # Simpler: stage 1 uses stride 2 on all dims (4->8, 6->12, 6->12),
        # stage 2 stride (4, 8, 8) via kernel-stride pairs to hit
        # (32, 96, 96).
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
        """Return synthetic MRI volumes of shape (B, out_channels, *out_shape)."""
        assert biomarker_embedding.dim() == 2 and biomarker_embedding.size(1) == self.embed_dim, \
            f"expected (B, {self.embed_dim}) got {tuple(biomarker_embedding.shape)}"
        B = biomarker_embedding.size(0)
        sd, sh, sw = self.seed_shape
        seed = self.seed_linear(biomarker_embedding).view(B, self.base_channels, sd, sh, sw)
        x = self.upsample1(seed)      # (B, C, 8, 12, 12)
        x = self.upsample2(x)          # (B, C, 32, 96, 96)
        x = self.refine(x)             # (B, out_channels, 32, 96, 96)

        # Resize to the exact out_shape in case stride multiplication was
        # off for an unusual seed configuration.
        if tuple(x.shape[-3:]) != self.out_shape:
            x = torch.nn.functional.interpolate(x, size=self.out_shape,
                                                mode="trilinear",
                                                align_corners=False)
        return x
