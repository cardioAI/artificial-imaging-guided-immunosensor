
from __future__ import annotations

import torch.nn as nn

class EDDDecoder(nn.Sequential):

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
