
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):

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
