from functools import reduce
import math
from typing import List, Tuple
import torch


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def lcm_list(xs: List[int]) -> int:
    return reduce(lcm, xs, 1)

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


__all__ = [
    "lcm",
    "lcm_list",
    "_find_multiple",
    "trunc_normal_init_",
]