"""
models.similarity
=================

Pure-function helpers for the similarity math that underpins both the
contrastive alignment loss (Figure 1b) and biomarker-to-image retrieval
(Figure 1f). Keeping these free functions lets downstream code (training
loops, clustering pipelines, retrieval scripts) call them directly without
dragging a whole class along.

Contents
--------
* ``cosine_similarity_matrix(a, b)`` -- row-wise L2 normalisation + outer
  product; returns the (N_a, N_b) cosine similarity matrix.
* ``temperature_scaled_similarity(a, b, temperature)`` -- cosine similarity
  divided by a temperature scalar; the input to InfoNCE cross-entropy.
* ``top_k_cosine_retrieval(query, database, k)`` -- cosine similarity from
  ``query`` (shape ``(N_q, D)`` or ``(D,)``) into a ``(N_db, D)`` database,
  returning the ``top_k`` indices and scores. Used by patient-78..108
  artificial-image retrieval.
* ``l2_normalize(x, eps)`` -- thin wrapper around ``F.normalize`` kept here
  so callers can import a single entry point for all similarity ops.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalisation along the last dimension."""
    return F.normalize(x, p=2, dim=-1, eps=eps)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``(N_a, N_b)`` cosine similarity between rows of ``a`` and ``b``."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.transpose(-1, -2)


def temperature_scaled_similarity(a: torch.Tensor,
                                  b: torch.Tensor,
                                  temperature: float = 0.15) -> torch.Tensor:
    """Cosine similarity / ``temperature``; feeds symmetric InfoNCE loss."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return cosine_similarity_matrix(a, b) / temperature


def top_k_cosine_retrieval(query: torch.Tensor,
                           database: torch.Tensor,
                           k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve the top-``k`` most similar rows from ``database`` for each query.

    Args:
        query:    ``(D,)`` or ``(N_q, D)`` tensor.
        database: ``(N_db, D)`` tensor of candidate embeddings.
        k:        number of neighbours to return; clipped to ``N_db``.

    Returns:
        ``(indices, scores)`` both shaped ``(N_q, k)`` (or ``(k,)`` when the
        query was a single vector). Scores are cosine similarities in [-1, 1].
    """
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
