"""
retrieval.biomarker_to_image
============================

Biomarker-to-image retrieval (Figure 1f). Given a biomarker embedding and a
database of image embeddings, return the indices and cosine similarities of
the top-k nearest image embeddings. For the held-out independent test cohort
(patients 78-108 by default), retrieval substitutes a synthetic image
embedding for the real one so the downstream clustering evaluates the
artificial imaging-guided strategy.

Contents
--------
* ``retrieve_image_from_biomarker`` -- the core retrieval function
  (cosine similarity + top-k). Re-exported from
  :class:`models.CardioAIModel`.
* ``extract_embeddings`` -- end-to-end extraction over the full 108-patient
  cohort. Applies the real image encoder to the 1-77 contrastive train+val
  cohort (57 + 20) and the retrieval path to the 78-108 test cohort (31),
  fusing each pair through :class:`models.CrossModalFusion`. Re-exported
  from :class:`clustering.pipeline.CardioAIClusteringAnalyzer`.
"""

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
    """Bidirectional top-k recall + mean reciprocal rank (manuscript eval).

    Args:
        biomarker_embeddings: (N, D) tensor.
        image_embeddings: (N, D) tensor. Row ``i`` must be the paired image
            embedding for biomarker row ``i``.
        k_values: iterable of k values for top-k recall.

    Returns:
        Dict with ``b2i_recall@k``, ``i2b_recall@k`` for each k plus
        ``mean_reciprocal_rank`` averaged over both retrieval directions.
    """
    biomarker_embeddings = F.normalize(biomarker_embeddings, p=2, dim=1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    n = biomarker_embeddings.size(0)
    sim = biomarker_embeddings @ image_embeddings.T  # (N, N)

    def _ranks(matrix: torch.Tensor) -> torch.Tensor:
        # ranks[i] = position of the paired column in sorted(desc) row i.
        diag = torch.diagonal(matrix)
        order = torch.argsort(matrix, dim=1, descending=True)
        positions = (order == torch.arange(n, device=matrix.device).unsqueeze(1)).nonzero(as_tuple=False)
        # positions[:, 0] = row, positions[:, 1] = rank (0-based)
        ranks = torch.zeros(n, dtype=torch.long, device=matrix.device)
        ranks[positions[:, 0]] = positions[:, 1]
        return ranks + 1  # 1-based

    b2i_ranks = _ranks(sim)          # biomarker -> image
    i2b_ranks = _ranks(sim.T)         # image -> biomarker

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
