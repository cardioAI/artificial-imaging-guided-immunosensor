"""
models.cardio_model
===================

Top-level CardioAI model that wires the individual Figure 1 components
together: biomarker encoder (1d), image encoder (1e), contrastive loss (1b)
and cross-modal fusion (1g). Also hosts the bidirectional retrieval helper
that realises Figure 1f.

Contents
--------
* ``CardioAIModel`` -- ``forward(biomarkers, images)`` returns a dict with
  the contrastive loss, the fused embedding, metrics, and optionally the
  per-modality embeddings. Convenience methods ``encode_biomarkers`` and
  ``encode_images`` expose the encoders directly.
  ``retrieve_image_from_biomarker`` and ``retrieve_biomarker_from_image``
  implement the bidirectional retrieval of Figure 1f via cosine similarity
  + top-k.
* ``create_model`` -- factory function applying sensible defaults; accepts
  an optional config dictionary for overrides.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .biomarker_encoder import BiomarkerEncoder
from .contrastive_decoder import ContrastiveDecoder
from .contrastive_loss import (ContrastiveLoss, HardNegativeInfoNCE,
                               RetrievalMSELoss, embedding_regularization)
from .cross_modal_fusion import CrossModalFusion
from .image_encoder import ImageEncoder


class CardioAIModel(nn.Module):
    """Complete Figure 1 model: encoders + contrastive loss + fusion + retrieval."""

    # Manuscript eq 3 weights for the combined training objective:
    # L_total = alpha*L_contrastive + beta*L_hard-negative
    #        + gamma*L_reconstruction + delta*L_regularization
    LOSS_WEIGHTS = {"alpha": 1.0, "beta": 0.3, "gamma": 0.1, "delta": 0.05}

    def __init__(self,
                 biomarker_input_dim: int = 6,
                 image_input_shape: Tuple[int, int, int] = (32, 96, 96),
                 embed_dim: int = 256,
                 fusion_dim: int = 256,
                 dropout: float = 0.1,
                 temperature: float = 0.07,
                 theta_hard: float = 0.6,
                 retrieval_beta: float = 5.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

        self.biomarker_encoder = BiomarkerEncoder(
            input_dim=biomarker_input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.image_encoder = ImageEncoder(
            input_shape=image_input_shape,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.hard_negative_loss = HardNegativeInfoNCE(temperature=temperature,
                                                     theta_hard=theta_hard)
        self.retrieval_loss = RetrievalMSELoss(beta=retrieval_beta)
        self.fusion = CrossModalFusion(embed_dim=embed_dim, fusion_dim=fusion_dim)
        # Trained contrastive decoder (manuscript: artificial images are
        # produced by "projecting biomarker embeddings into image space via
        # a trained contrastive decoder network").
        self.contrastive_decoder = ContrastiveDecoder(
            embed_dim=embed_dim,
            out_channels=4,
            out_shape=image_input_shape,
        )

    def forward(self,
                biomarkers: torch.Tensor,
                images: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        biomarker_embedding = self.biomarker_encoder(biomarkers)
        image_embedding = self.image_encoder(images)

        contrastive_loss, metrics = self.contrastive_loss(biomarker_embedding, image_embedding)
        hard_negative_loss = self.hard_negative_loss(biomarker_embedding, image_embedding)
        # Manuscript eq 15: retrieval MSE in embedding space.
        retrieval_mse_loss = self.retrieval_loss(biomarker_embedding, image_embedding)
        # Trained contrastive decoder: pixel-space reconstruction of the
        # real MRI volume from the biomarker embedding. Summed with the
        # embedding-space retrieval MSE so both obligations are reflected
        # in the L_reconstruction term (eq 3, gamma-weighted).
        decoded_images = self.contrastive_decoder(biomarker_embedding)
        decoder_reconstruction_loss = F.mse_loss(decoded_images, images)
        reconstruction_loss = retrieval_mse_loss + decoder_reconstruction_loss
        regularization_loss = embedding_regularization(biomarker_embedding, image_embedding)

        w = self.LOSS_WEIGHTS
        total_loss = (w["alpha"] * contrastive_loss
                      + w["beta"] * hard_negative_loss
                      + w["gamma"] * reconstruction_loss
                      + w["delta"] * regularization_loss)

        fused_embedding = self.fusion(biomarker_embedding, image_embedding)

        output = {
            "contrastive_loss": contrastive_loss,
            "hard_negative_loss": hard_negative_loss,
            "reconstruction_loss": reconstruction_loss,
            "regularization_loss": regularization_loss,
            "total_loss": total_loss,
            "fused_embedding": fused_embedding,
            "metrics": metrics,
        }
        if return_embeddings:
            output.update({
                "biomarker_embedding": biomarker_embedding,
                "image_embedding": image_embedding,
            })
        return output

    def set_temperature(self, temperature: float) -> None:
        """Mutate the contrastive / hard-negative temperature in place."""
        t = float(temperature)
        self.contrastive_loss.temperature.fill_(t)
        self.hard_negative_loss.temperature.fill_(t)

    def encode_biomarkers(self, biomarkers: torch.Tensor) -> torch.Tensor:
        return self.biomarker_encoder(biomarkers)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def decode_from_biomarker(self, biomarkers: torch.Tensor) -> torch.Tensor:
        """Project biomarker concentrations through the trained contrastive
        decoder to produce a synthetic MRI volume (manuscript Methods).

        Args:
            biomarkers: (B, 6) raw biomarker vector (same normalisation as
                training inputs).
        Returns:
            ``(B, 4, D, H, W)`` synthetic volumes.
        """
        embedding = self.biomarker_encoder(biomarkers)
        return self.contrastive_decoder(embedding)

    def retrieve_image_from_biomarker(self,
                                      biomarker_embedding: torch.Tensor,
                                      image_embeddings: torch.Tensor,
                                      top_k: int = 1
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Figure 1f retrieval: cosine similarity + top-k over an image-embedding database."""
        return self._cosine_topk(biomarker_embedding, image_embeddings, top_k)

    def retrieve_biomarker_from_image(self,
                                      image_embedding: torch.Tensor,
                                      biomarker_embeddings: torch.Tensor,
                                      top_k: int = 1
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Figure 1f mirror retrieval: image -> biomarker via cosine top-k."""
        return self._cosine_topk(image_embedding, biomarker_embeddings, top_k)

    @staticmethod
    def _cosine_topk(query: torch.Tensor,
                     database: torch.Tensor,
                     top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        top_k = min(top_k, database.size(0))
        similarities = F.cosine_similarity(query, database, dim=1)
        if top_k > 0:
            top_k_similarities, top_k_indices = torch.topk(similarities, top_k)
        else:
            top_k_similarities = torch.tensor([])
            top_k_indices = torch.tensor([])
        return top_k_indices, top_k_similarities


def create_model(config: Optional[Dict[str, Any]] = None) -> CardioAIModel:
    """Factory function producing a CardioAIModel with sensible defaults."""
    default_config = {
        "biomarker_input_dim": 6,
        "image_input_shape": (32, 96, 96),
        "embed_dim": 256,
        "fusion_dim": 256,
        "dropout": 0.1,
    }
    if config:
        default_config.update(config)
    return CardioAIModel(**default_config)
