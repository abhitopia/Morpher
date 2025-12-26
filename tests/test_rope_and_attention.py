# tests/test_rope_and_attention.py
import pytest
import torch

from modules import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from morpher import AttentionAdapter, AttnBackend


def _seed_all(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manual_rope_apply_rotate_half(x_bthd: torch.Tensor, cos_td: torch.Tensor, sin_td: torch.Tensor) -> torch.Tensor:
    """
    Matches your implementation:
      x_embed = x*cos + rotate_half(x)*sin
    where rotate_half splits last dim into halves and returns [-x2, x1].
    """
    orig_dtype = x_bthd.dtype
    x = x_bthd.to(cos_td.dtype)

    cos = cos_td.unsqueeze(-2)  # [T, 1, d]
    sin = sin_td.unsqueeze(-2)  # [T, 1, d]

    out = (x * cos) + (rotate_half(x) * sin)
    return out.to(orig_dtype)


@pytest.mark.parametrize("d", [8, 32, 64])
@pytest.mark.parametrize("max_pos", [16, 128])
def test_rotary_embedding_shapes_even_dim(d, max_pos):
    rope = RotaryEmbedding(dim=d, max_position_embeddings=max_pos, base=10000)
    cos, sin = rope()

    assert cos.shape == (max_pos, d)
    assert sin.shape == (max_pos, d)
    assert cos.dtype == sin.dtype
    assert cos.device == sin.device


@pytest.mark.parametrize("d", [8, 32])
def test_apply_rotary_matches_manual_reference_rotate_half(d):
    _seed_all(0)
    B, T, H = 2, 11, 5
    rope = RotaryEmbedding(dim=d, max_position_embeddings=64, base=10000)
    cos_full, sin_full = rope()
    cos = cos_full[:T]
    sin = sin_full[:T]

    q = torch.randn(B, T, H, d)
    k = torch.randn(B, T, H, d)

    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)

    q_ref = _manual_rope_apply_rotate_half(q, cos, sin)
    k_ref = _manual_rope_apply_rotate_half(k, cos, sin)

    assert torch.allclose(q2, q_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(k2, k_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("d", [8, 32, 64])
def test_rope_preserves_norms_and_dot_products_when_applied_to_both(d):
    _seed_all(0)
    B, T, H = 3, 19, 7
    rope = RotaryEmbedding(dim=d, max_position_embeddings=128, base=10000)
    cos_full, sin_full = rope()
    cos = cos_full[:T]
    sin = sin_full[:T]

    q = torch.randn(B, T, H, d)
    k = torch.randn(B, T, H, d)

    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)

    q_norm = torch.linalg.vector_norm(q, dim=-1)
    q2_norm = torch.linalg.vector_norm(q2, dim=-1)
    k_norm = torch.linalg.vector_norm(k, dim=-1)
    k2_norm = torch.linalg.vector_norm(k2, dim=-1)

    assert torch.allclose(q_norm, q2_norm, atol=1e-6, rtol=1e-5)
    assert torch.allclose(k_norm, k2_norm, atol=1e-6, rtol=1e-5)

    dot = (q * k).sum(dim=-1)
    dot2 = (q2 * k2).sum(dim=-1)
    assert torch.allclose(dot, dot2, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("d", [8, 32])
def test_rope_pos_offset_slicing_equivalence(d):
    _seed_all(0)
    B, H = 2, 4
    pos_offset = 17
    T = 13
    L = pos_offset + T + 5

    rope = RotaryEmbedding(dim=d, max_position_embeddings=L, base=10000)
    cos_full, sin_full = rope()

    q_big = torch.randn(B, L, H, d)
    k_big = torch.randn(B, L, H, d)

    q_big2, k_big2 = apply_rotary_pos_emb(q_big, k_big, cos_full[:L], sin_full[:L])

    q_win = q_big[:, pos_offset : pos_offset + T].clone()
    k_win = k_big[:, pos_offset : pos_offset + T].clone()
    cos = cos_full[pos_offset : pos_offset + T]
    sin = sin_full[pos_offset : pos_offset + T]
    q_win2, k_win2 = apply_rotary_pos_emb(q_win, k_win, cos, sin)

    assert torch.allclose(q_big2[:, pos_offset : pos_offset + T], q_win2, atol=1e-6, rtol=1e-5)
    assert torch.allclose(k_big2[:, pos_offset : pos_offset + T], k_win2, atol=1e-6, rtol=1e-5)


def test_apply_rotary_preserves_input_dtype():
    _seed_all(0)
    d = 32
    B, T, H = 1, 7, 2
    rope = RotaryEmbedding(dim=d, max_position_embeddings=32, base=10000)
    cos, sin = rope()
    cos = cos[:T]
    sin = sin[:T]

    q = torch.randn(B, T, H, d, dtype=torch.float32)
    k = torch.randn(B, T, H, d, dtype=torch.float32)
    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
    assert q2.dtype == q.dtype
    assert k2.dtype == k.dtype


def test_rope_odd_dim_raises_assertion_error_on_init():
    """
    RotaryEmbedding enforces even dim and raises AssertionError during initialization
    when given an odd dimension.
    """
    _seed_all(0)
    d = 7  # odd dimension
    with pytest.raises(AssertionError, match="RotaryEmbedding requires even dim"):
        RotaryEmbedding(dim=d, max_position_embeddings=16, base=10000)


def test_rope_pos_offset_overflow_raises_in_attention_adapter():
    _seed_all(0)
    B, T, K, N, d = 2, 16, 2, 3, 8
    max_pos = 20

    rope = RotaryEmbedding(dim=d, max_position_embeddings=max_pos, base=10000)
    attn = AttentionAdapter(backend=AttnBackend.SDPA, rope=rope).eval()

    qkv = torch.randn(B, T, K, N, 3 * d)

    # pos_offset too large => cos/sin slice shorter than T => should error downstream
    with pytest.raises((RuntimeError, AssertionError)):
        _ = attn(qkv, is_causal=True, dropout_p=0.0, pos_offset=max_pos - T + 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash_attn tests")
def test_flash3_matches_sdpa_no_dropout():
    try:
        import flash_attn  # noqa: F401
    except Exception:
        pytest.skip("flash_attn not importable")

    B, T, K, N, d = 2, 64, 2, 4, 32
    rope = RotaryEmbedding(dim=d, max_position_embeddings=256, base=10000)

    sdpa = AttentionAdapter(AttnBackend.SDPA, rope=rope).cuda().eval()
    fl3 = AttentionAdapter(AttnBackend.FLASH3, rope=rope).cuda().eval()

    qkv = torch.randn(B, T, K, N, 3*d, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        y_sdpa = sdpa(qkv, is_causal=True, dropout_p=0.0)
        y_fl3 = fl3(qkv, is_causal=True, dropout_p=0.0)

    # tolerances for fp16
    assert torch.allclose(y_sdpa, y_fl3, atol=2e-2, rtol=2e-2)
