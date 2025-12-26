import torch
import torch.nn as nn


class NoParamRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (simplified version without learnable weight).
    Faster than LayerNorm (no mean centering) and often performs similarly.
    Based on HRM's implementation and: https://arxiv.org/abs/1910.07467
    
    This version matches HRM's design: just normalization without a learnable scale parameter.
    The scale can be learned through residual connections and subsequent layers.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [..., *normalized_shape]
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute RMS: sqrt(mean(x^2)) over normalized dimensions
        # Normalize over the last len(normalized_shape) dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))
        variance = x.square().mean(dim=dims, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        return x.to(input_dtype)
