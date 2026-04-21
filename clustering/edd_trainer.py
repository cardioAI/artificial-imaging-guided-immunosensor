"""
clustering.edd_trainer
======================

Training loop for the Figure 1h encoder-decoder-discriminator-classifier
network. Runs jointly:

* Reconstruction loss (MSE between the original fused embedding and the
  decoder output).
* Adversarial loss (discriminator BCE on real vs reconstructed).
* MASLD classification loss (BCE-with-logits on the classifier head using
  the ``Label`` column aligned with the fused-embedding ordering).

After training, dumps per-patient prediction probabilities (``nafld_classification.csv``)
and a summary row (``nafld_classification_summary.csv``) with accuracy, AUC
and confusion-matrix counts.

Contents
--------
The training driver lives on
:class:`clustering.pipeline.CardioAIClusteringAnalyzer` and is re-exported
here:

* ``train_encoder_decoder_discriminator(num_epochs, batch_size, cls_weight)``
  -- trains the EDD and writes the classifier outputs.
"""

from .pipeline import CardioAIClusteringAnalyzer

train_encoder_decoder_discriminator = \
    CardioAIClusteringAnalyzer.train_encoder_decoder_discriminator

__all__ = ["train_encoder_decoder_discriminator"]
