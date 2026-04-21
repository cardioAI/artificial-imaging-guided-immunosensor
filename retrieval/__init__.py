"""
retrieval subpackage.

Implements the Figure 1f biomarker-to-image retrieval operation used both
(a) at training time by :mod:`training.trainer` via
:class:`models.CardioAIModel` and (b) at inference time by the clustering
pipeline to build artificial image embeddings for the independent test
cohort (patients 78-108 by default), matching the study protocol where the
57 / 20 / 31 split reserves 31 patients for artificial imaging-guided
clustering.
"""

from .biomarker_to_image import (
    extract_embeddings,
    retrieve_image_from_biomarker,
)

__all__ = ["extract_embeddings", "retrieve_image_from_biomarker"]
