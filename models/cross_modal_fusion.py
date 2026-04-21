"""
models.cross_modal_fusion
=========================

Cross-modal fusion module (Figure 1g). Takes a paired biomarker embedding
and image embedding (each in R^embed_dim) and produces a single fused
representation that feeds clustering, grouping, and the downstream EDD
framework (Figure 1h).

Contents
--------
* ``CrossModalFusion`` -- per-modality linear projection + concatenation +
  MLP + self-attention + LayerNorm. Output shape ``(B, fusion_dim)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """Fuse biomarker + image embeddings into a single embedding (Figure 1g)."""

    def __init__(self, embed_dim: int = 256, fusion_dim: int = 256):
        super().__init__()

        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

        self.biomarker_proj = nn.Linear(embed_dim, fusion_dim // 2)
        self.image_proj = nn.Linear(embed_dim, fusion_dim // 2)

        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self,
                biomarker_embedding: torch.Tensor,
                image_embedding: torch.Tensor) -> torch.Tensor:
        biomarker_proj = self.biomarker_proj(biomarker_embedding)
        image_proj = self.image_proj(image_embedding)
        fused = torch.cat([biomarker_proj, image_proj], dim=1)

        fused = self.fusion_layers(fused)

        fused_seq = fused.unsqueeze(1)
        attn_out, _ = self.attention(fused_seq, fused_seq, fused_seq)
        fused = fused + attn_out.squeeze(1)

        fused_embedding = self.norm(fused)
        return fused_embedding
