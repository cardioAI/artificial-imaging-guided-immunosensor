"""
clustering.classical_clustering
===============================

Classical "biomarkers only" clustering pipeline covering Figure 4c (negative
hierarchical heatmap), Figure 4e (PCA), Figure 4f (t-SNE) and Figure 4g
(UMAP). Uses z-score normalised biomarker vectors directly -- no image
information, no AI-enhanced embedding.

Contents
--------
Methods live on :class:`clustering.pipeline.CardioAIClusteringAnalyzer`
and are re-exported here:

* ``perform_classical_clustering`` -- top-level entry point; runs
  hierarchical + PCA + t-SNE + UMAP on the biomarker matrix.
* ``classical_hierarchical_clustering`` -- dendrogram + clustered heatmap
  (blue-white-orange colormap) for either the biomarker-only or AI-enhanced
  data (the ``prefix`` argument switches between Figure 4c and Figure 4d).
* ``classical_dimred_clustering`` -- PCA / t-SNE / UMAP scatter plots,
  one file per method. Used for both six-biomarker (Figure 4e-4g) and
  imaging-guided (Figure 4i-4k) clustering.
"""

from .pipeline import CardioAIClusteringAnalyzer

perform_classical_clustering = CardioAIClusteringAnalyzer.perform_classical_clustering
classical_hierarchical_clustering = CardioAIClusteringAnalyzer.classical_hierarchical_clustering
classical_dimred_clustering = CardioAIClusteringAnalyzer.classical_dimred_clustering

__all__ = [
    "perform_classical_clustering",
    "classical_hierarchical_clustering",
    "classical_dimred_clustering",
]
