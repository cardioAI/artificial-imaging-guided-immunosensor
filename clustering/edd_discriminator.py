
from __future__ import annotations

import torch.nn as nn

class EDDDiscriminator(nn.Sequential):

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.input_dim = input_dim
