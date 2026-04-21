"""
clustering.advanced_clustering
==============================

AI-enhanced clustering pipeline covering Figure 4d (positive hierarchical
heatmap), Figure 4i (PCA), Figure 4j (t-SNE) and Figure 4k (UMAP). Operates
on the refined latent embeddings produced by the encoder-decoder-
discriminator-classifier framework (Figure 1h), which in turn are built
from:

* contrastive train+val cohort (patients 1-77 by default, 57 + 20): real
  image embeddings fused with biomarker embeddings
* independent test cohort (patients 78-108 by default, 31): retrieved
  artificial image embeddings fused with biomarker embeddings
  (biomarker-to-image retrieval, Figure 1f)

Contents
--------
Methods live on :class:`clustering.pipeline.CardioAIClusteringAnalyzer`
and are re-exported here:

* ``perform_advanced_clustering`` -- top-level entry point; runs
  hierarchical + PCA + t-SNE + UMAP on the EDD-refined embeddings.
"""

from .pipeline import CardioAIClusteringAnalyzer

perform_advanced_clustering = CardioAIClusteringAnalyzer.perform_advanced_clustering

__all__ = ["perform_advanced_clustering"]
