"""
models subpackage.

Public API re-exported here so callers can write::

    from models import CardioAIModel, BiomarkerEncoder, ImageEncoder

without having to know which submodule each class lives in. Each submodule
is a focused, single-concern building block of the Figure 1 architecture.
"""

from .attention import MultiHeadSelfAttention
from .transformer_block import TransformerBlock
from .patch_modules import PatchEmbed3D, PatchMerging3D
from .biomarker_encoder import BiomarkerEncoder
from .image_encoder import ImageEncoder
from .cross_modal_fusion import CrossModalFusion
from .contrastive_loss import (ContrastiveLoss, HardNegativeInfoNCE,
                               RetrievalMSELoss)
from .contrastive_decoder import ContrastiveDecoder
from .cardio_model import CardioAIModel, create_model
from .projection_head import ProjectionHead
from .similarity import (
    l2_normalize,
    cosine_similarity_matrix,
    temperature_scaled_similarity,
    top_k_cosine_retrieval,
)

__all__ = [
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "PatchEmbed3D",
    "PatchMerging3D",
    "BiomarkerEncoder",
    "ImageEncoder",
    "CrossModalFusion",
    "ContrastiveLoss",
    "HardNegativeInfoNCE",
    "RetrievalMSELoss",
    "ContrastiveDecoder",
    "CardioAIModel",
    "create_model",
    "ProjectionHead",
    "l2_normalize",
    "cosine_similarity_matrix",
    "temperature_scaled_similarity",
    "top_k_cosine_retrieval",
]
