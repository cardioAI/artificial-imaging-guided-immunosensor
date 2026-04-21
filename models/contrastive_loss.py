"""
models.contrastive_loss
=======================

InfoNCE-style symmetric contrastive loss (Figure 1b, manuscript eq 1) and
hard-negative mined variant (eq 2) for biomarker / MRI alignment.

Contents
--------
* ``ContrastiveLoss`` -- symmetric InfoNCE with temperature tau = 0.07 by
  default. Returns ``(loss, metrics)``; metrics track per-direction
  cross-entropy and top-1 retrieval accuracy.
* ``HardNegativeInfoNCE`` -- hard-negative mined InfoNCE (manuscript eq 2)
  restricted to negatives exceeding theta_hard = 0.6 biomarker similarity.
* ``RetrievalMSELoss`` -- within-batch retrieval MSE (manuscript eq 15)
  between the real image embedding and a similarity-weighted average of
  other patients' image embeddings using temperature beta = 5.0
  (manuscript eq 14).
* ``embedding_regularization`` -- scalar penalty keeping projection-head
  outputs from drifting away from the unit sphere.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss for biomarker <-> image alignment (eq 1)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # Wrap tau in a buffer so annealing schedulers can mutate it in
        # place during training without detaching it from the optimiser.
        self.register_buffer("temperature",
                             torch.tensor(float(temperature)),
                             persistent=False)

    def forward(self,
                biomarker_embeddings: torch.Tensor,
                image_embeddings: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = biomarker_embeddings.size(0)

        biomarker_embeddings = F.normalize(biomarker_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        similarity_matrix = (biomarker_embeddings @ image_embeddings.T) / self.temperature
        labels = torch.arange(batch_size, dtype=torch.long, device=biomarker_embeddings.device)

        biomarker_to_image_loss = F.cross_entropy(similarity_matrix, labels)
        image_to_biomarker_loss = F.cross_entropy(similarity_matrix.T, labels)
        contrastive_loss = (biomarker_to_image_loss + image_to_biomarker_loss) / 2.0

        with torch.no_grad():
            biomarker_to_image_pred = similarity_matrix.argmax(dim=1)
            image_to_biomarker_pred = similarity_matrix.T.argmax(dim=1)
            biomarker_to_image_acc = (biomarker_to_image_pred == labels).float().mean()
            image_to_biomarker_acc = (image_to_biomarker_pred == labels).float().mean()
            accuracy = (biomarker_to_image_acc + image_to_biomarker_acc) / 2.0

        metrics = {
            "biomarker_to_image_loss": biomarker_to_image_loss,
            "image_to_biomarker_loss": image_to_biomarker_loss,
            "accuracy": accuracy,
            "biomarker_to_image_acc": biomarker_to_image_acc,
            "image_to_biomarker_acc": image_to_biomarker_acc,
        }
        return contrastive_loss, metrics


class HardNegativeInfoNCE(nn.Module):
    """Hard-negative mined InfoNCE (manuscript eq 2).

    Negatives ``j != i`` are admitted to the denominator only when their
    biomarker-image cosine similarity exceeds ``theta_hard``. The numerator
    always contains the positive pair so the loss is well-defined even when
    no hard negatives are present in the batch (in which case it degenerates
    to a constant, yielding zero gradient from this term).
    """

    def __init__(self, temperature: float = 0.07, theta_hard: float = 0.6):
        super().__init__()
        self.register_buffer("temperature",
                             torch.tensor(float(temperature)),
                             persistent=False)
        self.theta_hard = theta_hard

    def forward(self,
                biomarker_embeddings: torch.Tensor,
                image_embeddings: torch.Tensor
                ) -> torch.Tensor:
        batch_size = biomarker_embeddings.size(0)
        if batch_size < 2:
            return biomarker_embeddings.new_tensor(0.0)

        biomarker_embeddings = F.normalize(biomarker_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        # Raw cosine similarity matrix (pre-temperature) for hard-negative
        # gating. Diagonal entries are positive pairs and are always kept.
        sim = biomarker_embeddings @ image_embeddings.T
        hard_mask = (sim >= self.theta_hard)
        eye = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        hard_mask = hard_mask | eye  # always include positive

        # Scale by temperature for the softmax-style contrast.
        logits = sim / self.temperature

        # For each row i, build log-softmax over columns j in hard_mask_i.
        # Rather than doing row-by-row gather (which explodes in Python for
        # large batches), we add a large negative value to masked-out entries
        # so they vanish from the softmax.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(hard_mask, logits, torch.full_like(logits, neg_inf))

        # Positive-pair log prob = diag(logits) - logsumexp(masked_row)
        log_denom = torch.logsumexp(masked_logits, dim=1)
        pos_logits = torch.diagonal(logits)
        loss = -(pos_logits - log_denom).mean()
        return loss


class RetrievalMSELoss(nn.Module):
    """Within-batch retrieval MSE (manuscript eq 14 + eq 15).

    For each sample ``i`` we form an artificial image embedding from the
    other ``N-1`` samples in the batch using biomarker-space cosine
    similarity softmaxed with temperature ``beta = 5.0``, and minimize the
    MSE to the real image embedding. Requires a batch size of at least two.
    """

    def __init__(self, beta: float = 5.0):
        super().__init__()
        self.beta = beta

    def forward(self,
                biomarker_embeddings: torch.Tensor,
                image_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = biomarker_embeddings.size(0)
        if batch_size < 2:
            return biomarker_embeddings.new_tensor(0.0)

        bio_n = F.normalize(biomarker_embeddings, p=2, dim=1)
        # Biomarker-space cosine similarity between sample i and every other sample j.
        sim_bb = bio_n @ bio_n.T
        # Exclude self-pair by masking the diagonal with -inf.
        neg_inf = torch.finfo(sim_bb.dtype).min
        eye = torch.eye(batch_size, dtype=torch.bool, device=sim_bb.device)
        sim_bb = sim_bb.masked_fill(eye, neg_inf)
        # Softmax weights with temperature beta (eq 14).
        weights = F.softmax(self.beta * sim_bb, dim=1)
        # Artificial image embedding per row.
        z_artificial = weights @ image_embeddings
        # MSE on the embeddings (eq 15).
        return F.mse_loss(z_artificial, image_embeddings)


def embedding_regularization(*embeddings: torch.Tensor) -> torch.Tensor:
    """Soft L2 regulariser keeping unit-normed embeddings near the unit sphere.

    For embeddings already passed through ``F.normalize`` this evaluates to
    zero, but the term is retained so downstream consumers can observe the
    weighted sum in ``L_total`` (manuscript eq 3) explicitly.
    """
    if not embeddings:
        return torch.zeros((), dtype=torch.float32)
    pieces = [((e.norm(p=2, dim=-1) - 1.0) ** 2).mean() for e in embeddings]
    return torch.stack(pieces).mean()
