
# tests/test_global_slot.py
import pytest
import torch
import torch.nn as nn
from morpher import Morpher, StreamHeadAssignment, HeadInputScope, AttnBackend, CrossStreamAttention

def _make_model(use_global_slot=True, global_slot_scale=4, d=8, N=4):
    """Factory for tests."""
    # time_scales=[1, 2, 4] -> N=4 (lcm)
    return Morpher(
        io_dim=16,
        d=d,
        time_scales=[1, 2, 4],
        enc_dec_rank=4,
        use_global_slot=use_global_slot,
        global_slot_scale=global_slot_scale,
        attn_backend=AttnBackend.SDPA,
    )

def test_cross_stream_attention_shapes():
    """Verify CrossStreamAttention handles tensor shapes correctly."""
    B, T, N, D = 2, 5, 4, 32
    d = 8
    
    # Init module
    csa = CrossStreamAttention(
        stream_dim=D,
        attn_dim=16,
        out_dim=d
    )
    
    z = torch.randn(B, T, N, D)
    out = csa(z)
    
    # Output should be [B, T, N, out_dim]
    assert out.shape == (B, T, N, d)

def test_global_slot_disabled_by_default():
    """Ensure global slot is off by default and params match."""
    model = Morpher(
        io_dim=16,
        d=8,
        time_scales=[1, 2, 4],
        enc_dec_rank=4,
        # Default use_global_slot=False
    )
    assert not model.use_global_slot
    # S = 1+2+4 = 7
    assert model.S == 7
    assert model.D == 7 * 8 

def test_global_slot_enabled_increases_dimensions():
    """Ensure enabling global slot increases S by 1."""
    model = _make_model(use_global_slot=True, d=8, global_slot_scale=2)
    
    assert model.use_global_slot
    assert model.global_slot_scale == 2
    # S = 7 + 1 = 8
    assert model.S == 8
    assert model.D == 8 * 8
    assert hasattr(model, 'cross_stream_attn')
    assert hasattr(model, 'ln_global')

def test_forward_runs_with_global_slot():
    """Smoke test for forward pass."""
    model = _make_model(use_global_slot=True)
    B, T = 2, 10
    x = torch.randn(B, T, 16)
    
    # Should run without error
    y = model(x, R=1, is_causal=True)
    assert y.shape == (B, T, 16)

def test_global_slot_updates_at_correct_frequency():
    """
    Verify global slot only changes at multiples of global_slot_scale.
    We'll monkey-patch _update_global_slot to track calls.
    """
    model = _make_model(use_global_slot=True, global_slot_scale=2)
    
    # Store history of when it was called
    call_times = []
    
    original_update = model._update_global_slot
    
    def tracked_update(z):
        # We can't easily see 't' here since it's passed to step() not _update_global_slot directly,
        # but we can verify it *is* called.
        # Actually, let's just use a counter in the outer scope
        # But wait, step() is called inside a loop in forward_cycle
        return original_update(z)

    # Instead of monkeypatching, let's inspect the slot values.
    # We can manually run step()
    
    z = torch.randn(2, 1, model.N, model.D)
    
    # t=0: Should update global slot (0 % 2 == 0)
    # We need to spy on the values.
    # The global slot is the LAST slot.
    
    # Let's mock the cross_stream_attn to return a fixed distinct value 
    # so we can see if it was written.
    fixed_val = 100.0
    
    # Mock forward of cross_stream_attn
    def mock_attn_forward(x):
        return torch.full((x.shape[0], x.shape[1], x.shape[2], model.d), fixed_val)
    
    model.cross_stream_attn.forward = mock_attn_forward
    
    # t=1: Should NOT update (1 % 2 != 0)
    z_out_t1 = model.step(z.clone(), t=1, is_causal=True)
    
    # Check global slot (last slot)
    z_slots_t1 = z_out_t1.view(2, 1, model.N, model.S, model.d)
    global_slot_t1 = z_slots_t1[..., -1, :]
    
    # Should NOT be fixed_val (should be roughly same as input z, maybe drifted by mixer/mixer dropout?)
    # Wait, mixer updates ALL slots. So z changes every step.
    # The *Global Slot Update* step (step 6) happens *after* mixer.
    # But if step 6 didn't run, the global slot would just be whatever the mixer output was.
    # If step 6 ran, it would have the residual + fixed_val.
    
    # This is tricky because mixer modifies everything.
    # Let's disable mixer for this test?
    model.mixer = nn.Identity()
    model.ln_mixer = nn.Identity()
    model.dropout_mixer = nn.Identity()
    
    # Also disable regular attention update to isolate global slot?
    # Regular attention updates active slots. 
    # Global slot is scale N, does it ever get updated by regular scheduler?
    # Our time_scales are [1, 2, 4]. Global slot index is 7.
    # Indices: s=1 -> [0], s=2 -> [1,2], s=4 -> [3,4,5,6].
    # So slot 7 is NEVER updated by regular schedule.
    # BUT mixer updates everything.
    # Since we disabled mixer, slot 7 should be unchanged if global update doesn't run.
    
    # t=1 (no update expected)
    z_in = torch.zeros_like(z)
    z_out_t1 = model.step(z_in, t=1, is_causal=True)
    z_slots_t1 = z_out_t1.view(2, 1, model.N, model.S, model.d)
    assert torch.all(z_slots_t1[..., -1, :] == 0), "Global slot changed at t=1 but shouldn't have"
    
    # t=0 (update expected)
    # with our mock, it adds fixed_val
    z_out_t0 = model.step(z_in, t=0, is_causal=True)
    z_slots_t0 = z_out_t0.view(2, 1, model.N, model.S, model.d)
    
    # The value should be roughly fixed_val (plus residual which was 0)
    # Note: _update_global_slot adds residual: new = old + update
    # old was 0. update is 100. new should be 100.
    assert torch.allclose(z_slots_t0[..., -1, :], torch.tensor(fixed_val)), "Global slot didn't update at t=0"

def test_gradient_flow_global_slot():
    """Verify gradients flow through global slot attention."""
    model = _make_model(use_global_slot=True, global_slot_scale=1) # Update every step
    B, T = 2, 5
    x = torch.randn(B, T, 16, requires_grad=True)
    
    y = model(x, R=1, is_causal=True)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    # Verify CSA weights got grads
    assert model.cross_stream_attn.W_qkv.weight.grad is not None


def test_global_slot_isolation_from_per_scale_attention():
    """
    Verify that regular per-scale attention NEVER selects the global slot.
    The global slot (index S-1) should not be in active_slots for any phase.
    """
    model = _make_model(use_global_slot=True)
    
    # time_scales = [1, 2, 4] gives active slots per phase:
    # For each phase, active_slots_k contains indices into [0, S-1]
    # The global slot is at index S-1 = 7
    # Regular scales use indices 0-6 (sum of 1+2+4 = 7 slots, indices 0-6)
    # So global slot index 7 should NEVER appear in active_slots
    
    global_slot_idx = model.global_slot_idx  # Should be 7
    assert global_slot_idx == model.S - 1
    
    # Check all phases
    for phase in range(model.N):
        active_slots = model.tables.active_slots[phase]  # [K]
        assert global_slot_idx not in active_slots.tolist(), \
            f"Global slot {global_slot_idx} found in active_slots at phase {phase}"


def test_cross_stream_attention_different_dimensions():
    """Test CrossStreamAttention with attn_dim != out_dim."""
    B, T, N, D = 2, 5, 4, 64
    attn_dim = 16  # Q/K dimension
    out_dim = 32   # V/output dimension (different!)
    
    csa = CrossStreamAttention(
        stream_dim=D,
        attn_dim=attn_dim,
        out_dim=out_dim,
        dropout=0.0,
    )
    
    z = torch.randn(B, T, N, D)
    out = csa(z)
    
    # Output should be [B, T, N, out_dim], not attn_dim
    assert out.shape == (B, T, N, out_dim)
    
    # Verify W_qkv has correct shape: [2*attn_dim + out_dim, stream_dim]
    expected_out_features = 2 * attn_dim + out_dim  # 16 + 16 + 32 = 64
    assert csa.W_qkv.weight.shape == (expected_out_features, D)


def test_global_slot_accumulation_across_updates():
    """
    Verify global slot properly accumulates updates.
    Each _update_global_slot call should add to the residual.
    """
    model = _make_model(use_global_slot=True, global_slot_scale=1, d=8)
    
    B, T = 1, 1
    z = torch.zeros(B, T, model.N, model.D)
    
    # Mock cross_stream_attn to return constant value
    increment = 1.0
    def mock_forward(x):
        return torch.full((x.shape[0], x.shape[1], x.shape[2], model.d), increment)
    model.cross_stream_attn.forward = mock_forward
    
    # Call _update_global_slot directly 3 times
    num_updates = 3
    for _ in range(num_updates):
        z = model._update_global_slot(z)
    
    # Global slot should have accumulated: 0 + 1 + 1 + 1 = 3
    z_slots = z.view(B, T, model.N, model.S, model.d)
    global_slot = z_slots[..., -1, :]
    
    expected_val = float(num_updates) * increment
    assert torch.allclose(global_slot, torch.tensor(expected_val)), \
        f"Expected global slot = {expected_val}, got {global_slot.mean().item()}"
