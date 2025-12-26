import torch.nn as nn
from torch.nn import functional as F

from modules.casted_linear import CastedLinear
from utils import _find_multiple


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
