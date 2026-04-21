"""
clustering.chord_diagrams
=========================

Figure 4a and Figure 4b chord diagrams of within-group biomarker
correlations. Drawn with pure matplotlib (no pycirclize dependency): six
biomarker nodes equispaced on a circle, quadratic Bezier chords encoding
pairwise correlations, node size encoding z-score deviation from the cohort
mean, chord colour encoding correlation sign (positive = orange,
negative = blue).

Contents
--------
Methods live on :class:`clustering.pipeline.CardioAIClusteringAnalyzer`
and are re-exported here for discoverability and direct imports:

* ``create_chord_diagrams`` -- renders Figure 4a (negative group) and
  Figure 4b (positive group). Writes PNG / EPS / TIFF for each.
* ``draw_chord_diagram`` -- underlying single-diagram renderer used by
  ``create_chord_diagrams`` for one group.
"""

from .pipeline import CardioAIClusteringAnalyzer

create_chord_diagrams = CardioAIClusteringAnalyzer.create_chord_diagrams
draw_chord_diagram = CardioAIClusteringAnalyzer._draw_chord_diagram

__all__ = ["create_chord_diagrams", "draw_chord_diagram"]
