"""
models.projection_head
======================

Reusable projection head used at the top of both CardioAI encoders. Maps
a hidden representation into the shared embedding space, optionally with
L2 normalisation so downstream cosine-based operations behave well.

Design:
    in_dim -> hidden_dim -> out_dim
    Configurable activation between the two Linear layers (GELU for the
    image encoder, ReLU for the biomarker encoder in the current code).

Contents
--------
* ``ProjectionHead`` -- small ``nn.Module`` with a ``normalize`` flag.
  Useful when you want to attach a contrastive-ready head onto any new
  backbone without reimplementing the same pattern.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection into the shared embedding space."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: Optional[int] = None,
                 activation: str = "gelu",
                 normalize: bool = True):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else out_dim

        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(),
        }
        if activation not in act_map:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Expected one of {sorted(act_map)}."
            )

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_map[activation],
            nn.Linear(hidden_dim, out_dim),
        )
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x
