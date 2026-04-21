"""
clustering subpackage.

Top-level re-exports so ``from clustering import CardioAIClusteringAnalyzer``
works. Individual concerns are accessible via the focused submodules:
:mod:`clustering.chord_diagrams`, :mod:`clustering.classical_clustering`,
:mod:`clustering.advanced_clustering`, :mod:`clustering.clustering_metrics`,
:mod:`clustering.edd_trainer`, :mod:`clustering.edd_model`.
"""

from .pipeline import (
    CardioAIClusteringAnalyzer,
    EncoderDecoderDiscriminator,
)
from .edd_encoder import EDDEncoder
from .edd_decoder import EDDDecoder
from .edd_discriminator import EDDDiscriminator
from .edd_classifier import EDDClassifier

__all__ = [
    "CardioAIClusteringAnalyzer",
    "EncoderDecoderDiscriminator",
    "EDDEncoder",
    "EDDDecoder",
    "EDDDiscriminator",
    "EDDClassifier",
]
