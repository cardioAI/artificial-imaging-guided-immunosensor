
from typing import Dict

import torch
import torch.nn.functional as F

from models import CardioAIModel
from clustering.pipeline import CardioAIClusteringAnalyzer

retrieve_image_from_biomarker = CardioAIModel.retrieve_image_from_biomarker
extract_embeddings = CardioAIClusteringAnalyzer.extract_embeddings

def evaluate_retrieval(biomarker_embeddings: torch.Tensor,
                       image_embeddings: torch.Tensor,
                       k_values=(1, 5, 10)) -> Dict[str, float]:
    biomarker_embeddings = F.normalize(biomarker_embeddings, p=2, dim=1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    n = biomarker_embeddings.size(0)
    sim = biomarker_embeddings @ image_embeddings.T

    def _ranks(matrix: torch.Tensor) -> torch.Tensor:

        diag = torch.diagonal(matrix)
        order = torch.argsort(matrix, dim=1, descending=True)
        positions = (order == torch.arange(n, device=matrix.device).unsqueeze(1)).nonzero(as_tuple=False)

        ranks = torch.zeros(n, dtype=torch.long, device=matrix.device)
        ranks[positions[:, 0]] = positions[:, 1]
        return ranks + 1

    b2i_ranks = _ranks(sim)
    i2b_ranks = _ranks(sim.T)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f'b2i_recall@{k}'] = float((b2i_ranks <= k).float().mean().item())
        metrics[f'i2b_recall@{k}'] = float((i2b_ranks <= k).float().mean().item())
    b2i_mrr = float((1.0 / b2i_ranks.float()).mean().item())
    i2b_mrr = float((1.0 / i2b_ranks.float()).mean().item())
    metrics['b2i_mean_reciprocal_rank'] = b2i_mrr
    metrics['i2b_mean_reciprocal_rank'] = i2b_mrr
    metrics['mean_reciprocal_rank'] = 0.5 * (b2i_mrr + i2b_mrr)
    return metrics

__all__ = [
    "retrieve_image_from_biomarker",
    "extract_embeddings",
    "evaluate_retrieval",
]
