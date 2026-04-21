"""
clustering.edd_model
====================

The encoder-decoder-discriminator-classifier network of Figure 1h. Trained
on the fused biomarker+image embeddings to produce:

1. A refined latent embedding (for clustering and grouping).
2. A reconstruction decoder (for representation quality).
3. A discriminator (adversarial signal on reconstruction quality).
4. A binary classifier head (MASLD Negative:0 vs Positive:1), matching the
   explicit 0/1 output of Figure 1h.

Contents
--------
* ``EncoderDecoderDiscriminator`` -- the nn.Module itself. Methods:
    - ``encode(x)``, ``decode(z)`` -- the autoencoder halves.
    - ``discriminate(x)`` -- sigmoid real/fake score.
    - ``classify(z)`` -- MASLD logit (apply sigmoid for probability).
    - ``forward(x)`` -- returns all of the above in a dict with keys
      ``latent``, ``reconstructed``, ``real_score``, ``fake_score``, ``logit``.
"""

from .pipeline import EncoderDecoderDiscriminator

__all__ = ["EncoderDecoderDiscriminator"]
