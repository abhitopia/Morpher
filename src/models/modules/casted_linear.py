import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import trunc_normal_init_


class CastedLinear(nn.Module):
    """
    Linear layer that uses truncated normal initialization and explicit casting (no need for Autocast).
    Taken from HRM's implementation.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features, )))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_init_(self.weight, std=1.0 / (self.weight.shape[1] ** 0.5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # HRM's approach: explicit casting for mixed precision compatibility
        # Alternative: rely on autocast (comment out .to(dtype) calls and test)
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
