"""
clustering.clustering_metrics
=============================

Quantitative comparison of classical vs AI-enhanced clustering. Produces
Figure 4h (silhouette score) and Figure 4l (Calinski-Harabasz index).

Contents
--------
Methods live on :class:`clustering.pipeline.CardioAIClusteringAnalyzer`
and are re-exported here:

* ``evaluate_clustering`` -- runs both metrics on classical and advanced
  embedding matrices, writes ``evaluation_results.json`` and the two bar
  charts. Returns a dict with ``classical``, ``advanced`` and
  ``improvement`` keys.
* ``compute_clustering_scores`` -- stateless helper returning the pair of
  metrics for a single ``(data, labels)`` input.
* ``create_evaluation_plots`` -- renders the NI (blue) vs AINI (orange)
  bar charts for Figure 4h and Figure 4l.
"""

from .pipeline import CardioAIClusteringAnalyzer

evaluate_clustering = CardioAIClusteringAnalyzer.evaluate_clustering
compute_clustering_scores = CardioAIClusteringAnalyzer.compute_clustering_scores
create_evaluation_plots = CardioAIClusteringAnalyzer.create_evaluation_plots

__all__ = [
    "evaluate_clustering",
    "compute_clustering_scores",
    "create_evaluation_plots",
]
