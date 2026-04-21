"""
clustering.edd_encoder
======================

Encoder half of the EDD framework (Figure 1h, left wing). Maps a 512-d
fused biomarker+image embedding into a compact latent code suitable for
clustering, grouping and downstream classification.

Design:
    input_dim -> hidden_dim -> hidden_dim/2 -> latent_dim
    ReLU + Dropout between Linear layers
    Final Tanh on the latent so values stay bounded in [-1, 1].

Contents
--------
* ``EDDEncoder`` -- ``nn.Sequential`` subclass so it plugs into
  :class:`clustering.pipeline.EncoderDecoderDiscriminator` without
  perturbing existing ``state_dict`` keys (keys remain
  ``encoder.0.weight``, ``encoder.2.weight``, ...).
"""

from __future__ import annotations

import torch.nn as nn


class EDDEncoder(nn.Sequential):
    """Fused-embedding -> latent encoder (Figure 1h, left)."""

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh(),
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
