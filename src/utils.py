import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # Based on HRM's implementation
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor



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
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # HRM's approach: explicit casting for mixed precision compatibility
        # Alternative: rely on autocast (comment out .to(dtype) calls and test)
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


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