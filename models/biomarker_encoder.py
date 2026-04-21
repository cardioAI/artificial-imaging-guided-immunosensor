"""
models.biomarker_encoder
========================

Biomarker encoder implementing Figure 1d. Six plasma biomarker concentrations
(AST, ALT, GGT, CTSD, CK18, FGF21) are mapped into the shared embedding space
used for contrastive alignment with 3D MRI embeddings.

Figure 1d dataflow
------------------
Biomarkers -> MLP -> LayerNorm -> Multi-head self-attention -> LayerNorm
           -> (+ MLP_out) -> Embedding

The ``+`` symbol in the figure sums the MLP output with the output of the
*second* LayerNorm (post-norm residual), i.e.

    embedding = MLP_out + LN2(MHSA(LN1(MLP_out)))

not ``LN2(MLP_out + MHSA(LN1(MLP_out)))``.

Contents
--------
* ``BiomarkerEncoder`` -- configurable hidden dim / embedding dim / heads /
  dropout; outputs are L2-normalised for direct use in InfoNCE-style
  contrastive losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention


class BiomarkerEncoder(nn.Module):
    """Six-biomarker encoder (Figure 1d)."""

    def __init__(self,
                 input_dim: int = 6,
                 hidden_dim: int = 128,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Manuscript eq 5: W_proj -> GELU -> LayerNorm. The second linear
        # stage inside the MLP block uses GELU as well, matching the
        # activation family stated in the Methods text.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.embedding_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, biomarkers: torch.Tensor) -> torch.Tensor:
        assert biomarkers.dim() == 2 and biomarkers.size(1) == self.input_dim, \
            f"Expected biomarkers shape (batch_size, {self.input_dim}), got {biomarkers.shape}"

        mlp_out = self.mlp(biomarkers).unsqueeze(1)

        x = self.layer_norm1(mlp_out)
        x = self.attention(x)
        x = self.layer_norm2(x)
        x = mlp_out + x

        x = x.squeeze(1)
        biomarker_embedding = self.embedding_proj(x)
        biomarker_embedding = F.normalize(biomarker_embedding, p=2, dim=1)
        return biomarker_embedding
