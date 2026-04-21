"""
clustering.edd_classifier
=========================

Binary MASLD classifier head (Figure 1h, bottom-right). Operates on the
latent embedding produced by :class:`clustering.edd_encoder.EDDEncoder`
and emits a single pre-sigmoid logit that maps to MASLD Negative:0 vs
Positive:1. Trained with BCE-with-logits against the ``Label`` column.

Design:
    latent_dim -> classifier_hidden -> 1
    GELU + Dropout between the two Linear layers
    NO sigmoid here -- the loss (``BCEWithLogitsLoss``) applies it; apply
    ``torch.sigmoid`` manually for probabilities at inference time.

Contents
--------
* ``EDDClassifier`` -- ``nn.Sequential`` subclass; drop-in replacement for
  the inline classifier head in
  :class:`clustering.pipeline.EncoderDecoderDiscriminator`.
"""

from __future__ import annotations

import torch.nn as nn


class EDDClassifier(nn.Sequential):
    """Latent -> MASLD logit classifier (Figure 1h, bottom-right)."""

    def __init__(self,
                 latent_dim: int = 128,
                 classifier_hidden: int = 64,
                 dropout: float = 0.25):
        super().__init__(
            nn.Linear(latent_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )
        self.latent_dim = latent_dim
