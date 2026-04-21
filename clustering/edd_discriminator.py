"""
clustering.edd_discriminator
============================

Adversarial discriminator of the EDD framework (Figure 1h, vertical bar
on the right). Classifies an input embedding as "real" (sampled from the
fused-embedding distribution) vs "reconstructed" (output of the encoder-
decoder pair). Its gradient pushes the decoder to produce reconstructions
indistinguishable from real embeddings, improving latent quality.

Design:
    input_dim -> hidden_dim -> hidden_dim/2 -> 1
    LeakyReLU(0.2) + Dropout between Linear layers
    Final Sigmoid for a [0, 1] real-vs-fake score.

Contents
--------
* ``EDDDiscriminator`` -- ``nn.Sequential`` subclass; drop-in replacement
  for the inline discriminator.
"""

from __future__ import annotations

import torch.nn as nn


class EDDDiscriminator(nn.Sequential):
    """Real vs reconstructed embedding discriminator (Figure 1h)."""

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.input_dim = input_dim
