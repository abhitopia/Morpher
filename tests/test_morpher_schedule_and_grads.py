import pytest
import torch
import torch.nn as nn

from morpher import (
    Morpher,
    PhaseTables,
    StreamHeadAssignment,
    HeadInputScope,
    AttnBackend,
)

# ----------------------------
# Helpers
# ----------------------------
def _seed_all(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _offsets(time_scales):
    offs = []
    cur = 0
    for s in time_scales:
        offs.append(cur)
        cur += s
    return offs

def _expected_active_slots(time_scales):
    time_scales = sorted(time_scales)
    K = len(time_scales)
    N = 1
    for s in time_scales:
        N = (N * s) // torch.gcd(torch.tensor(N), torch.tensor(s)).item()

    offs = _offsets(time_scales)
    active = torch.empty(N, K, dtype=torch.long)
    for phase in range(N):
        for k, s in enumerate(time_scales):
            active[phase, k] = offs[k] + (phase % s)
    return active  # [N, K]

def _make_tiny_model(
    *,
    io_dim=16,
    d=8,
    time_scales=(1,2,4),
    enc_dec_rank=4,
    assignment=StreamHeadAssignment.PRIVATE,
    scope=HeadInputScope.SLOT,
    use_rope=False,
):
    # dropout=0 and eval mode => deterministic
    m = Morpher(
        io_dim=io_dim,
        d=d,
        time_scales=list(time_scales),
        enc_dec_rank=enc_dec_rank,
        mixer_expansion=2.0,
        stream_head_assignment=assignment,
        head_input_scope=scope,
        attn_backend=AttnBackend.SDPA,
        dropout=0.0,
        use_rope=use_rope,
        max_position_embeddings=64,
    ).eval()
    return m

def _zero_module_params(mod: nn.Module):
    for p in mod.parameters(recurse=True):
        p.data.zero_()

# ----------------------------
# Dummy components to isolate schedule/writeback
# ----------------------------
class DummyProjector(nn.Module):
    """Returns zeros with correct packed QKV shape [B,T,K,N,3d]."""
    def __init__(self, K, N, d):
        super().__init__()
        self.K, self.N, self.d = K, N, d

    def project_slot(self, x_btknd, *, phase, stream_to_head, head_to_stream):
        B,T,K,N,_ = x_btknd.shape
        assert (K, N) == (self.K, self.N)
        return torch.zeros(B, T, K, N, 3*self.d, device=x_btknd.device, dtype=x_btknd.dtype)

    def project_stream(self, x_btni, *, phase, stream_to_head, head_to_stream):
        B,T,N,_ = x_btni.shape
        return torch.zeros(B, T, self.K, N, 3*self.d, device=x_btni.device, dtype=x_btni.dtype)

class DummyAttnAddOne(nn.Module):
    """Ignores qkv and returns ones [B,T,K,N,d] so active slots increment by +1."""
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, qkv_btk_n3d, *, is_causal, dropout_p, pos_offset=0):
        B,T,K,N,three_d = qkv_btk_n3d.shape
        d = three_d // 3
        assert d == self.d
        return torch.ones(B, T, K, N, d, device=qkv_btk_n3d.device, dtype=qkv_btk_n3d.dtype)

# ----------------------------
# 1) Schedule table correctness
# ----------------------------
@pytest.mark.parametrize("time_scales", [(1,2,4), (1,3), (1,2,3), (1,3,5)])
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
def test_active_slots_matches_closed_form(time_scales, assignment):
    tables = PhaseTables(list(time_scales), assignment)
    expected = _expected_active_slots(list(time_scales))
    assert torch.equal(tables.active_slots.cpu(), expected.cpu())

@pytest.mark.parametrize("time_scales", [(1,2,4), (1,3,5), (1,2,3,6)])
def test_active_slots_are_disjoint_and_size_k(time_scales):
    # disjointness should hold because each scale has its own contiguous slot block
    time_scales = list(sorted(time_scales))
    expected = _expected_active_slots(time_scales)  # [N,K]
    N,K = expected.shape
    for phase in range(N):
        slots = expected[phase].tolist()
        assert len(slots) == K
        assert len(set(slots)) == K, f"phase {phase} had duplicate active slots: {slots}"

@pytest.mark.parametrize("time_scales", [(1,2,4), (1,3,5), (1,2,3,6)])
def test_each_slot_activated_equally_often_over_full_cycle(time_scales):
    # Over phases 0..N-1, for each scale s, each of its s slots is active exactly N/s times.
    time_scales = list(sorted(time_scales))
    active = _expected_active_slots(time_scales)  # [N,K]
    N,K = active.shape
    offs = _offsets(time_scales)

    for k, s in enumerate(time_scales):
        # slot indices for this scale block
        block = list(range(offs[k], offs[k] + s))
        counts = {j: 0 for j in block}
        for phase in range(N):
            j = int(active[phase, k].item())
            counts[j] += 1
        expected = N // s
        assert all(c == expected for c in counts.values()), (k, s, counts, expected)

# ----------------------------
# 2) Step follows schedule (writes only active slots) across configs
# ----------------------------
@pytest.mark.parametrize("time_scales", [(1,2,4), (1,3,5)])
@pytest.mark.parametrize("assignment", [StreamHeadAssignment.PRIVATE, StreamHeadAssignment.SHARED])
@pytest.mark.parametrize("scope", [HeadInputScope.SLOT, HeadInputScope.SCALES, HeadInputScope.STREAM])
def test_step_updates_exactly_active_slots(time_scales, assignment, scope):
    _seed_all(0)
    m = _make_tiny_model(
        io_dim=12,
        d=6,
        time_scales=time_scales,
        enc_dec_rank=4,
        assignment=assignment,
        scope=scope,
        use_rope=False,
    )

    # Ensure mixer doesn't touch non-active slots (otherwise *everything* changes)
    _zero_module_params(m.mixer)

    # Isolate schedule/writeback: projector=0, attn=+1
    m.projector = DummyProjector(m.K, m.N, m.d)
    m.attn = DummyAttnAddOne(m.d)

    B,T = 2,3
    z = torch.randn(B, T, m.N, m.D)
    t = 7
    phase = t % m.N
    active_k = m.tables.active_slots[phase]  # [K]

    z_next = m.step(z, t=t, is_causal=True)

    z_slots = z.view(B, T, m.N, m.S, m.d)
    z_next_slots = z_next.view(B, T, m.N, m.S, m.d)

    # Determine which slot indices changed at all (over B,T,N,d)
    delta = (z_next_slots - z_slots).abs().amax(dim=(0,1,2,4))  # [S]
    changed = (delta > 0).nonzero(as_tuple=False).flatten().tolist()

    expected_changed = sorted([int(i) for i in active_k.tolist()])
    assert sorted(changed) == expected_changed, (changed, expected_changed)

    # And ensure the change is exactly +1 everywhere in the changed slots
    for s in expected_changed:
        diff = z_next_slots[..., s, :] - z_slots[..., s, :]
        assert torch.allclose(diff, torch.ones_like(diff))

# ----------------------------
# 3) Full-cycle schedule “counting” test
#    Start from zeros, each active slot increments by +1 at each micro-step.
#    After one full cycle (N steps), slot value == number of times it was active.
# ----------------------------
@pytest.mark.parametrize("time_scales", [(1,2,4), (1,3,5), (1,2,3,6)])
@pytest.mark.parametrize("cycles", [1, 2])
def test_cycle_slot_update_counts_match_theory(time_scales, cycles):
    _seed_all(0)
    m = _make_tiny_model(
        io_dim=8,
        d=4,
        time_scales=time_scales,
        enc_dec_rank=2,
        assignment=StreamHeadAssignment.PRIVATE,
        scope=HeadInputScope.SLOT,
        use_rope=False,
    )
    _zero_module_params(m.mixer)
    m.projector = DummyProjector(m.K, m.N, m.d)
    m.attn = DummyAttnAddOne(m.d)

    B,T = 1,1
    z = torch.zeros(B, T, m.N, m.D)

    # run cycles*N micro-steps
    t0 = 0
    for _ in range(cycles * m.N):
        z = m.step(z, t=t0, is_causal=True)
        t0 += 1

    z_slots = z.view(B, T, m.N, m.S, m.d)

    # Expected count per slot depends on its scale: each slot in scale s block increments N/s per cycle
    time_scales = list(sorted(time_scales))
    offs = _offsets(time_scales)
    for k, s in enumerate(time_scales):
        expected = cycles * (m.N // s)
        for j in range(offs[k], offs[k] + s):
            block = z_slots[..., j, :]  # [B,T,N,d]
            target = torch.full_like(block, float(expected))
            assert torch.allclose(block, target), f"slot {j} expected {expected}, got {block.flatten()[0].item()}"

# ----------------------------
# 4) Gradient-graft correctness (core)
#    Compare:
#      (A) actual forward (with burn-in + graft) gradient dL/dz0 captured from encoder output
#      (B) reference gradient by starting from a leaf z_start == burned-in state and unrolling only grad_cycles
#    They must match. Then verify encoder parameter grads match chain rule.
# ----------------------------
def test_gradient_graft_matches_reference_and_encoder_chain_rule():
    _seed_all(0)

    # Keep tiny for speed but nontrivial
    m = _make_tiny_model(
        io_dim=10,
        d=6,
        time_scales=(1,2,4),
        enc_dec_rank=4,
        assignment=StreamHeadAssignment.PRIVATE,
        scope=HeadInputScope.SLOT,
        use_rope=False,
    ).train()  # grads enabled

    B,T = 2,3
    x = torch.randn(B, T, 10)

    R = 3
    grad_cycles = 1
    burn_cycles = R - grad_cycles
    assert burn_cycles > 0

    # --- (A) Actual forward/backward: capture z0 and its grad ---
    captured = {}
    def enc_hook(mod, inp, out):
        out.retain_grad()
        captured["z0"] = out
        return out

    h = m.encoder.register_forward_hook(enc_hook)

    m.zero_grad(set_to_none=True)
    y = m(x, R=R, is_causal=True, grad_cycles=grad_cycles)
    loss = (y**2).mean()
    loss.backward()

    z0_A = captured["z0"]
    grad_z0_A = z0_A.grad.detach().clone()

    # store encoder param grads from actual run
    grads_enc_A = {}
    for name, p in m.encoder.named_parameters():
        if p.grad is not None:
            grads_enc_A[name] = p.grad.detach().clone()

    h.remove()

    # --- (B) Reference: compute z_burned (no grad), then treat it as a leaf and unroll grad_cycles only ---
    m.zero_grad(set_to_none=True)

    z0 = m.encoder(x)              # has grad
    z0.retain_grad()

    # burn-in from detached z0 (matching model.forward semantics)
    z = z0.detach()
    t = 0
    with torch.no_grad():
        for _ in range(burn_cycles):
            z = m.forward_cycle(z, t0=t, is_causal=True)
            t += m.N

    # leaf start state with burned-in *value*
    z_start = z.clone().detach().requires_grad_(True)

    z_run = z_start
    for _ in range(grad_cycles):
        z_run = m.forward_cycle(z_run, t0=t, is_causal=True)
        t += m.N

    y_ref = m.decoder(z_run)
    loss_ref = (y_ref**2).mean()
    loss_ref.backward()

    grad_z_start = z_start.grad.detach().clone()

    # 1) Graft correctness: dL/dz0 in actual == dL/dz_start in reference
    assert torch.allclose(grad_z0_A, grad_z_start, atol=1e-5, rtol=1e-4)

    # 2) Encoder chain rule: encoder param grads in actual == backprop z0 with upstream grad_z_start
    m.zero_grad(set_to_none=True)
    z0_2 = m.encoder(x)
    torch.autograd.backward(z0_2, grad_z_start)

    for name, p in m.encoder.named_parameters():
        if name in grads_enc_A:
            assert p.grad is not None
            assert torch.allclose(p.grad, grads_enc_A[name], atol=1e-5, rtol=1e-4), f"encoder grad mismatch: {name}"

# ----------------------------
# 5) “Grad depth knob” sanity:
#    grad_cycles=2 should generally change gradients vs grad_cycles=1 (same R)
# ----------------------------
@pytest.mark.parametrize("scope", [HeadInputScope.SLOT, HeadInputScope.SCALES, HeadInputScope.STREAM])
def test_grad_cycles_changes_gradients(scope):
    _seed_all(0)
    m = _make_tiny_model(
        io_dim=12,
        d=6,
        time_scales=(1,2,4),
        enc_dec_rank=4,
        assignment=StreamHeadAssignment.PRIVATE,
        scope=scope,
        use_rope=False,
    ).train()

    B,T = 2,3
    x = torch.randn(B, T, 12)

    def run(grad_cycles):
        m.zero_grad(set_to_none=True)
        y = m(x, R=3, is_causal=True, grad_cycles=grad_cycles)
        (y**2).mean().backward()
        # pick one stable param tensor to compare (projector weights are a good one)
        p = m.projector.Wqkv
        return p.grad.detach().clone()

    g1 = run(grad_cycles=1)
    g2 = run(grad_cycles=2)

    # They don't have to be wildly different, but should not be identical.
    assert not torch.allclose(g1, g2, atol=0.0, rtol=0.0)
    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
