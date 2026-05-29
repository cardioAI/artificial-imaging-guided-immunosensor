
from __future__ import annotations

import torch.nn as nn

class EDDClassifier(nn.Sequential):

    def __init__(self,
                 latent_dim: int = 128,
                 classifier_hidden: int = 64,
                 dropout: float = 0.25):
        super().__init__(
            nn.Linear(latent_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )
        self.latent_dim = latent_dim
