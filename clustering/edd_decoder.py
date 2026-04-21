"""
clustering.edd_decoder
======================

Decoder half of the EDD framework (Figure 1h, right wing). Reconstructs a
512-d fused embedding from the encoder's latent code. Reconstruction MSE
against the original embedding provides the primary self-supervised signal
during EDD training.

Design:
    latent_dim -> hidden_dim/2 -> hidden_dim -> input_dim
    ReLU + Dropout between Linear layers
    Final Tanh so the reconstructed embedding matches the encoder's
    bounded output convention.

Contents
--------
* ``EDDDecoder`` -- ``nn.Sequential`` subclass; drop-in replacement for the
  inline decoder previously defined inside
  :class:`clustering.pipeline.EncoderDecoderDiscriminator`.
"""

from __future__ import annotations

import torch.nn as nn


class EDDDecoder(nn.Sequential):
    """Latent -> reconstructed embedding decoder (Figure 1h, right)."""

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
