
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.transpose(-1, -2)

def temperature_scaled_similarity(a: torch.Tensor,
                                  b: torch.Tensor,
                                  temperature: float = 0.15) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return cosine_similarity_matrix(a, b) / temperature

def top_k_cosine_retrieval(query: torch.Tensor,
                           database: torch.Tensor,
                           k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    squeeze_out = False
    if query.dim() == 1:
        query = query.unsqueeze(0)
        squeeze_out = True
    k = max(0, min(k, database.size(0)))
    if k == 0:
        empty = query.new_empty((query.size(0), 0))
        return empty.long(), empty
    sims = cosine_similarity_matrix(query, database)
    scores, indices = torch.topk(sims, k=k, dim=-1)
    if squeeze_out:
        return indices.squeeze(0), scores.squeeze(0)
    return indices, scores
