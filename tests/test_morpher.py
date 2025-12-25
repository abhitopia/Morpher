import math
import pytest
import torch

# Import from your module
from morpher import (
    PhaseTables, StreamHeadAssignment,
    _route_streams_to_heads_btkni, _route_heads_to_streams_btkni,
    QKVProjector, HeadInputScope, AttnBackend, Morpher
)

def _seed_all(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# A) PhaseTables
# -------------------------
@pytest.mark.parametrize("time_scales", [[1,2,4], [1,3], [1,2,3,6]])
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
            # s2h maps stream->head; h2s maps head->stream
            s = torch.arange(N, device=s2h.device)
            heads = s2h[phase, k, s]
            back = h2s[phase, k, heads]
            assert torch.equal(back.cpu(), s.cpu())

# -------------------------
# B) Routing helpers invert
# -------------------------
def test_routing_inverses():
    _seed_all(0)
    B,T,K,N,in_dim = 2,3,4,5,6
    x = torch.randn(B,T,K,N,in_dim)
    # random permutation per k
    perms = torch.stack([torch.randperm(N) for _ in range(K)], dim=0)  # [K,N]
    # head_to_stream: for head h, which stream to read from -> perms
    h2s = perms
    # stream_to_head inverse
    s2h = torch.empty_like(h2s)
    for k in range(K):
        s2h[k, h2s[k]] = torch.arange(N)

    x_head = _route_streams_to_heads_btkni(x, h2s)
    x_back = _route_heads_to_streams_btkni(x_head, s2h)
    assert torch.allclose(x, x_back)

# -------------------------
# C) QKVProjector correctness
# -------------------------
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
def test_project_slot_matches_reference(assignment):
    _seed_all(0)
    B,T,K,N,d = 2,3,3,4,8
    proj = QKVProjector(K=K, N=N, d=d, in_dim=d, assignment=assignment)
    proj.reset_parameters()

    # dummy tables for routing in SHARED case
    time_scales = [1,2,4]  # N=4, K=3
    tables = PhaseTables(time_scales, assignment)

    x = torch.randn(B,T,K,N,d)
    phase = 1

    out = proj.project_slot(
        x, phase=phase,
        stream_to_head=tables.stream_to_head,
        head_to_stream=tables.head_to_stream,
    )  # [B,T,K,N,3d]

    if assignment == StreamHeadAssignment.PRIVATE:
        # reference via einsum: (btkn i) * (kn i o) -> btkn o
        ref = torch.einsum("btkni,knio->btkno", x, proj.Wqkv)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    else:
        # explicit "route -> project -> unroute" reference
        h2s = tables.head_to_stream[phase]  # [K,N]
        s2h = tables.stream_to_head[phase]  # [K,N]
        x_head = _route_streams_to_heads_btkni(x, h2s)
        ref_head = torch.einsum("btkni,knio->btkno", x_head, proj.Wqkv)
        ref = _route_heads_to_streams_btkni(ref_head, s2h)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)

def test_project_stream_vectorized_equals_loop_shared():
    _seed_all(0)
    B,T,N,in_dim,K,d = 2,3,4,16,3,8

    # Force SHARED
    proj_vec = QKVProjector(K=K, N=N, d=d, in_dim=in_dim, assignment=StreamHeadAssignment.SHARED,
                           max_shared_xhead_bytes=10**12)  # always vectorize
    proj_loop = QKVProjector(K=K, N=N, d=d, in_dim=in_dim, assignment=StreamHeadAssignment.SHARED,
                            max_shared_xhead_bytes=0)      # always loop

    proj_vec.reset_parameters()
    # share weights so we compare paths only
    proj_loop.Wqkv.data.copy_(proj_vec.Wqkv.data)

    tables = PhaseTables([1,2,4], StreamHeadAssignment.SHARED)
    x = torch.randn(B,T,N,in_dim)
    phase = 2

    y_vec = proj_vec.project_stream(x, phase=phase,
                                    stream_to_head=tables.stream_to_head,
                                    head_to_stream=tables.head_to_stream)
    y_loop = proj_loop.project_stream(x, phase=phase,
                                      stream_to_head=tables.stream_to_head,
                                      head_to_stream=tables.head_to_stream)
    assert torch.allclose(y_vec, y_loop, atol=1e-5, rtol=1e-4)

# -------------------------
# D) Only active slots updated (when mixer disabled)
# -------------------------
def _zero_module_params(mod: torch.nn.Module):
    for p in mod.parameters(recurse=True):
        p.data.zero_()

@pytest.mark.parametrize("scope", [HeadInputScope.SLOT, HeadInputScope.SCALES, HeadInputScope.STREAM])
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
def test_step_only_writes_active_slots_when_mixer_is_zero(scope, assignment):
    _seed_all(0)
    B,T,io_dim = 2,5,12
    time_scales = [1,2,4]
    d = 8

    m = Morpher(
        io_dim=io_dim, d=d, time_scales=time_scales, enc_dec_rank=4,
        mixer_expansion=4.0,
        stream_head_assignment=assignment,
        head_input_scope=scope,
        attn_backend=AttnBackend.SDPA,
        dropout=0.0,
        use_rope=False,
    ).eval()

    # disable mixer so z_mixed == z
    _zero_module_params(m.mixer)

    x = torch.randn(B,T,io_dim)
    z = m.encoder(x)  # [B,T,N,D]

    t = 3
    phase = t % m.N
    active_k = m.tables.active_slots[phase]  # [K]
    z_next = m.step(z, t=t, is_causal=True)

    z_slots = z.view(B,T,m.N,m.S,m.d)
    z_next_slots = z_next.view(B,T,m.N,m.S,m.d)

    # mask of non-active slots
    all_idx = torch.arange(m.S)
    non_active = all_idx[~torch.isin(all_idx, active_k)]

    # non-active slots must be identical
    assert torch.allclose(
        z_slots.index_select(3, non_active),
        z_next_slots.index_select(3, non_active),
        atol=0.0, rtol=0.0
    )

# -------------------------
# E/F/G) Smoke + grads + compile
# -------------------------
@pytest.mark.parametrize("scope", [HeadInputScope.SLOT, HeadInputScope.SCALES, HeadInputScope.STREAM])
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
def test_forward_runs_and_grads(scope, assignment):
    _seed_all(0)
    B,T,io_dim = 2,4,16
    time_scales = [1,2,4]
    d = 8

    m = Morpher(
        io_dim=io_dim, d=d, time_scales=time_scales, enc_dec_rank=4,
        mixer_expansion=4.0,
        stream_head_assignment=assignment,
        head_input_scope=scope,
        attn_backend=AttnBackend.SDPA,
        dropout=0.0,
        use_rope=True,
        max_position_embeddings=64,
    )

    x = torch.randn(B,T,io_dim, requires_grad=True)
    y = m(x, R=2, is_causal=True, grad_cycles=1)
    loss = (y**2).mean()
    loss.backward()

    # basic finiteness checks
    assert torch.isfinite(loss).item()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"non-finite grad in {name}"

@pytest.mark.parametrize("scope", [HeadInputScope.SLOT, HeadInputScope.SCALES, HeadInputScope.STREAM])
def test_compile_smoke(scope):
    _seed_all(0)
    B,T,io_dim = 2,4,16
    m = Morpher(
        io_dim=io_dim, d=8, time_scales=[1,2,4], enc_dec_rank=4,
        stream_head_assignment=StreamHeadAssignment.PRIVATE,
        head_input_scope=scope,
        attn_backend=AttnBackend.SDPA,
        dropout=0.0,
        use_rope=False,
    ).eval()

    x = torch.randn(B,T,io_dim)

    cm = torch.compile(m)  # inductor
    y = cm(x, R=2, is_causal=True, grad_cycles=1)
    assert y.shape == (B,T,io_dim)
