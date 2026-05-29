
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()

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

        sim = biomarker_embeddings @ image_embeddings.T
        hard_mask = (sim >= self.theta_hard)
        eye = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        hard_mask = hard_mask | eye

        logits = sim / self.temperature

        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(hard_mask, logits, torch.full_like(logits, neg_inf))

        log_denom = torch.logsumexp(masked_logits, dim=1)
        pos_logits = torch.diagonal(logits)
        loss = -(pos_logits - log_denom).mean()
        return loss

class RetrievalMSELoss(nn.Module):

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

        sim_bb = bio_n @ bio_n.T

        neg_inf = torch.finfo(sim_bb.dtype).min
        eye = torch.eye(batch_size, dtype=torch.bool, device=sim_bb.device)
        sim_bb = sim_bb.masked_fill(eye, neg_inf)

        weights = F.softmax(self.beta * sim_bb, dim=1)

        z_artificial = weights @ image_embeddings

        return F.mse_loss(z_artificial, image_embeddings)

def embedding_regularization(*embeddings: torch.Tensor) -> torch.Tensor:
    if not embeddings:
        return torch.zeros((), dtype=torch.float32)
    pieces = [((e.norm(p=2, dim=-1) - 1.0) ** 2).mean() for e in embeddings]
    return torch.stack(pieces).mean()
