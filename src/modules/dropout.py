import torch
import torch.nn as nn
from torch.nn import functional as F


class Dropout(nn.Module):
    """
    Dropout active only when:
      - module.training == True
      - torch.is_grad_enabled() == True (disabled under no_grad/inference_mode)
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        if not (self.training and torch.is_grad_enabled()):
            return x
        return F.dropout(x, p=self.p, training=True)
