import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def _find_multiple(a: int, b: int) -> int:
    """
    Find the smallest multiple of b that is >= a.

    This is commonly used to ensure tensor dimensions are multiples of certain values
    (e.g., multiples of 256 for GPU efficiency).

    Examples:
        _find_multiple(257, 256) -> 512
        _find_multiple(100, 32) -> 128
    """
    return (-(a // -b)) * b


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


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.

    A gated activation that combines Swish (SiLU) with gating, commonly used in
    modern transformer architectures like LLaMA and PaLM. Often performs better
    than GELU while being similarly efficient.

    Formula: SwiGLU(x) = (SiLU(xW_gate) * xW_up) @ W_down

    Args:
        hidden_size: Input and output dimension
        expansion: Expansion factor for the intermediate dimension.
                  Total intermediate size = round(expansion * hidden_size * 2/3)
                  The *2/3 ensures the total parameter count is similar to a
                  single linear layer with expansion factor.

    Based on: https://arxiv.org/abs/2002.05202 (GLU Variants Improve Transformer)
    """
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        # Calculate intermediate dimension, rounded up to multiple of 256 for efficiency
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        # Single projection for both gate and up projections (fused for efficiency)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using truncated normal (matches HRM approach)."""
        self.gate_up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(self, x):
        # Split the gate_up_proj output into gate and up components
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        # Apply SiLU to gate, multiply by up, then project down
        return self.down_proj(F.silu(gate) * up)