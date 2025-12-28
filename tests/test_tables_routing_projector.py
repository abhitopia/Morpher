# tests/test_tables_routing_projector.py
import pytest
import torch

from models.morpher import (
    PhaseTables,
    StreamHeadAssignment,
    _route_streams_to_heads_btkni,
    _route_heads_to_streams_btkni,
    QKVProjector,
)


def _seed_all(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# PhaseTables invariants
# -------------------------
@pytest.mark.parametrize("time_scales", [[1, 2, 4], [1, 3], [1, 2, 3, 6]])
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
def test_phase_tables_invariants(time_scales, assignment):
    tables = PhaseTables(time_scales, assignment)
    K, N = tables.K, tables.N
    S = sum(sorted(time_scales))

    active = tables.active_slots  # [N, K]
    assert active.shape == (N, K)
    assert active.min().item() >= 0
    assert active.max().item() < S

    # head_to_stream and stream_to_head inverses
    s2h = tables.stream_to_head  # [N, K, N]
    h2s = tables.head_to_stream  # [N, K, N]
    for phase in range(N):
        for k in range(K):
            s = torch.arange(N, device=s2h.device)
            heads = s2h[phase, k, s]
            back = h2s[phase, k, heads]
            assert torch.equal(back.cpu(), s.cpu())


# -------------------------
# Routing helpers invert
# -------------------------
def test_routing_inverses():
    _seed_all(0)
    B, T, K, N, in_dim = 2, 3, 4, 5, 6
    x = torch.randn(B, T, K, N, in_dim)

    perms = torch.stack([torch.randperm(N) for _ in range(K)], dim=0)  # [K,N]
    h2s = perms
    s2h = torch.empty_like(h2s)
    for k in range(K):
        s2h[k, h2s[k]] = torch.arange(N)

    x_head = _route_streams_to_heads_btkni(x, h2s)
    x_back = _route_heads_to_streams_btkni(x_head, s2h)
    assert torch.allclose(x, x_back)


# -------------------------
# QKVProjector correctness
# -------------------------
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
def test_project_slot_matches_reference(assignment):
    _seed_all(0)
    B, T, K, N, d = 2, 3, 3, 4, 8
    proj = QKVProjector(K=K, N=N, d=d, in_dim=d, assignment=assignment)
    proj.reset_parameters()

    # tables for routing
    tables = PhaseTables([1, 2, 4], assignment)

    x = torch.randn(B, T, K, N, d)
    phase = 1

    out = proj.project_slot(
        x,
        phase=phase,
        stream_to_head=tables.stream_to_head,
        head_to_stream=tables.head_to_stream,
    )  # [B,T,K,N,3d]

    if assignment == StreamHeadAssignment.PRIVATE:
        ref = torch.einsum("btkni,knio->btkno", x, proj.Wqkv)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    else:
        h2s = tables.head_to_stream[phase]  # [K,N]
        s2h = tables.stream_to_head[phase]  # [K,N]
        x_head = _route_streams_to_heads_btkni(x, h2s)
        ref_head = torch.einsum("btkni,knio->btkno", x_head, proj.Wqkv)
        ref = _route_heads_to_streams_btkni(ref_head, s2h)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_project_stream_vectorized_equals_loop_shared():
    _seed_all(0)
    B, T, N, in_dim, K, d = 2, 3, 4, 16, 3, 8

    proj_vec = QKVProjector(
        K=K,
        N=N,
        d=d,
        in_dim=in_dim,
        assignment=StreamHeadAssignment.SHARED,
        max_shared_xhead_bytes=10**12,  # always vectorize
    )
    proj_loop = QKVProjector(
        K=K,
        N=N,
        d=d,
        in_dim=in_dim,
        assignment=StreamHeadAssignment.SHARED,
        max_shared_xhead_bytes=0,  # always loop
    )

    proj_vec.reset_parameters()
    proj_loop.Wqkv.data.copy_(proj_vec.Wqkv.data)

    tables = PhaseTables([1, 2, 4], StreamHeadAssignment.SHARED)
    x = torch.randn(B, T, N, in_dim)
    phase = 2

    y_vec = proj_vec.project_stream(
        x,
        phase=phase,
        stream_to_head=tables.stream_to_head,
        head_to_stream=tables.head_to_stream,
    )
    y_loop = proj_loop.project_stream(
        x,
        phase=phase,
        stream_to_head=tables.stream_to_head,
        head_to_stream=tables.head_to_stream,
    )

    assert torch.allclose(y_vec, y_loop, atol=1e-5, rtol=1e-4)
