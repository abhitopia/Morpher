from __future__ import annotations

from typing import List
from enum import StrEnum
from math import gcd
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)

def lcm_list(xs: List[int]) -> int:
    return reduce(lcm, xs, 1)


class Dropout(nn.Module):
    """
    Dropout active only when:
      - module.training == True
      - torch.is_grad_enabled() == True (disabled under no_grad/inference_mode)
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        if not (self.training and torch.is_grad_enabled()):
            return x
        return F.dropout(x, p=self.p, training=True)


# -----------------------------
# Stream LoRA Encoder / Decoder
# -----------------------------
class StreamLoRAEncoder(nn.Module):
    """
    x:  [B, T, input_dim]
    z0: [B, T, N, output_dim]
    """
    def __init__(self, input_dim: int, output_dim: int, num_splits: int, rank: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = num_splits
        self.rank = rank

        self.enc_base = nn.Linear(input_dim, output_dim, bias=False)
        self.enc_A = nn.Linear(input_dim, rank, bias=False)

        # [N, r, D] is nicer for bmm
        self.enc_B = nn.Parameter(torch.zeros(rank, num_splits, output_dim))

        # start near-off but not exactly 0 (for gradient flow)
        self.enc_beta = nn.Parameter(torch.tensor(0.01))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.enc_base.weight)
        nn.init.xavier_uniform_(self.enc_A.weight)
        nn.init.zeros_(self.enc_B)
        with torch.no_grad():
            self.enc_beta.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  [B,T,input_dim]
        z0: [B,T,N,output_dim]
        """
        B, T, _ = x.shape

        base = self.enc_base(x)  # [B,T,D]
        h = self.enc_A(x)        # [B,T,r]

        # Compute delta via batched GEMM:
        #   For each stream n: delta[:,:,n,:] = h @ B_n
        # Reshape h to [B*T, r], then expand to [N, B*T, r] without copying
        h_bt = h.reshape(B * T, self.rank)                         # [BT, r]
        h_n = h_bt.unsqueeze(0).expand(self.N, B * T, self.rank)    # [N, BT, r]

        # enc_B: [N, r, D]
        delta_n = torch.bmm(h_n, self.enc_B)                        # [N, BT, D]
        delta = delta_n.permute(1, 0, 2).reshape(B, T, self.N, self.output_dim)  # [B,T,N,D]

        return base.unsqueeze(2) + self.enc_beta * delta



class StreamLoRADecoder(nn.Module):
    """
    z: [B, T, N, input_dim]
    y: [B, T, output_dim]
    """
    def __init__(
        self,
        num_splits: int,
        input_dim: int,
        output_dim: int,
        rank: int,
        use_layernorm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.N = num_splits
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        in_dim = self.N * self.input_dim
        self.ln = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.proj_down = nn.Linear(in_dim, rank, bias=bias)
        self.proj_up = nn.Linear(rank, output_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, N, D = z.shape
        assert N == self.N and D == self.input_dim
        cat = z.reshape(B, T, N * D)
        h = self.proj_down(self.ln(cat))
        return self.proj_up(h)


# -----------------------------
# Enums
# -----------------------------
class StreamHeadAssignment(StrEnum):
    SHARED = "shared"     # phase-permuted within scale
    PRIVATE = "private"   # per-stream head within scale (identity)

class HeadInputScope(StrEnum):
    SLOT = "slot"
    STREAM = "stream"
    SCALES = "scales"


# -----------------------------
# Morpher
# -----------------------------
class Morpher(nn.Module):
    def __init__(
        self,
        io_dim: int,
        d: int,
        time_scales: List[int],
        enc_dec_rank: int,
        mixer_hidden_dim: int | None = None,
        stream_head_assignment: StreamHeadAssignment = StreamHeadAssignment.SHARED,
        head_input_scope: HeadInputScope = HeadInputScope.SLOT,
        dropout: float = 0.0,
    ):
        super().__init__()
        time_scales = sorted(time_scales)
        assert time_scales[0] == 1, "include scale=1 as fastest"
        assert stream_head_assignment in StreamHeadAssignment
        assert head_input_scope in HeadInputScope

        self.time_scales = time_scales
        self.stream_head_assignment = stream_head_assignment
        self.head_input_scope = head_input_scope

        self.dropout_p = float(dropout)
        self.dropout_mixer = Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.dropout_attn = Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # K = number of scales updated per micro-step
        self.K = len(self.time_scales)
        self.d = d

        # N = number of streams / phase period
        self.N = lcm_list(self.time_scales)

        # S = number of slots per stream
        self.S = sum(self.time_scales)

        # D = stream state dim
        self.D = self.S * self.d

        # Encoder / Decoder
        self.encoder = StreamLoRAEncoder(io_dim, self.D, self.N, enc_dec_rank)
        self.decoder = StreamLoRADecoder(self.N, self.D, io_dim, enc_dec_rank)

        # Mixer
        self.mixer_hidden_dim = mixer_hidden_dim if mixer_hidden_dim is not None else 4 * self.D
        self.ln_mixer = nn.LayerNorm(self.D)
        self.mixer = nn.Sequential(
            nn.Linear(self.D, self.mixer_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mixer_hidden_dim, self.D),
        )

        # Attention input dims per scope
        if self.head_input_scope == HeadInputScope.SLOT:
            self.head_input_dim = self.d
        elif self.head_input_scope == HeadInputScope.SCALES:
            self.head_input_dim = self.K * self.d
        else:
            self.head_input_dim = self.D

        self.ln_attn = nn.LayerNorm(self.head_input_dim)

        # Fused Wqkv: [K, N, in_dim, 3d]
        self.Wqkv = nn.Parameter(torch.empty(self.K, self.N, self.head_input_dim, 3 * self.d))

        self._build_active_slots_table()
        self._build_permutation_tables()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wqkv.reshape(self.K * self.N, self.head_input_dim, 3 * self.d))
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    # -----------------------------
    # Tables
    # -----------------------------
    def _build_active_slots_table(self):
        active_slots = torch.empty(self.N, self.K, dtype=torch.long)
        for phase in range(self.N):
            slot_offset = 0
            for scale_idx, scale in enumerate(self.time_scales):
                active_slots[phase, scale_idx] = slot_offset + (phase % scale)
                slot_offset += scale
        self.register_buffer("active_slots", active_slots, persistent=False)  # [N,K]

    def _build_permutation_tables(self):
        """
        For each phase and scale:
          perm[phase, k, stream] = head_id within scale (0..N-1) used by that stream
          inv [phase, k, head]   = stream index that should feed that head
        PRIVATE => identity.
        """
        perm = torch.empty(self.N, self.K, self.N, dtype=torch.long)
        inv = torch.empty(self.N, self.K, self.N, dtype=torch.long)

        stream_ids = torch.arange(self.N, dtype=torch.long)
        for phase in range(self.N):
            for k, s in enumerate(self.time_scales):
                j = phase % s
                if self.stream_head_assignment == StreamHeadAssignment.PRIVATE:
                    p = stream_ids
                else:
                    alpha = self.N // s
                    p = (stream_ids + alpha * j) % self.N
                perm[phase, k] = p
                inv_p = torch.empty_like(p)
                inv_p[p] = stream_ids
                inv[phase, k] = inv_p

        self.register_buffer("perm_within_scale", perm, persistent=False)      # [N,K,N]
        self.register_buffer("invperm_within_scale", inv, persistent=False)   # [N,K,N]

    # -----------------------------
    # QKV projection helpers
    # -----------------------------
    def _project_qkv_slot_vectorized(self, x_slot: torch.Tensor, phase: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SLOT fast path (best when in_dim=d):
          x_slot: [B,T,N,K,d] after LN
        Returns:
          q,k,v: [B, N*K, T, d]
        """
        B, T, N, K, d = x_slot.shape
        assert N == self.N and K == self.K and d == self.d

        # [B,T,N,K,d] -> [B,T,K,N,d]
        x_in = x_slot.permute(0, 1, 3, 2, 4)  # [B,T,K,N,d]

        if self.stream_head_assignment == StreamHeadAssignment.PRIVATE:
            # Identity wiring: no permutations needed.
            # Head order == stream order.
            x_head = x_in  # [B,T,K,N,d]
            # Pack (K*N) heads as batch for bmm
            x_h = x_head.permute(2, 3, 0, 1, 4).reshape(K * N, B * T, d)          # [K*N, BT, d]
            w = self.Wqkv.reshape(K * N, d, 3 * d)                                # [K*N, d, 3d]
            qkv_h = torch.bmm(x_h, w)                                             # [K*N, BT, 3d]
            qkv = qkv_h.reshape(K, N, B, T, 3 * d).permute(2, 3, 0, 1, 4)         # [B,T,K,N,3d]
            q, k, v = qkv.split(d, dim=-1)                                        # each [B,T,K,N,d]
            # Pack to SDPA head axis: [B,T,K,N,d] -> [B,N,K,T,d] -> [B,N*K,T,d]
            q = q.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            k = k.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            v = v.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            return q, k, v

        # SHARED wiring: permute streams -> heads and back
        p = self.perm_within_scale[phase]      # [K,N] stream->head
        inv = self.invperm_within_scale[phase] # [K,N] head->stream

        # gather streams into HEAD order along dim=3 (stream axis)
        inv_idx = inv.view(1, 1, K, N, 1).expand(B, T, K, N, d)
        x_head = torch.gather(x_in, dim=3, index=inv_idx)  # [B,T,K,N,d] head order

        # Project all (K*N) heads in one batched GEMM
        x_h = x_head.permute(2, 3, 0, 1, 4).reshape(K * N, B * T, d)        # [K*N, BT, d]
        w = self.Wqkv.reshape(K * N, d, 3 * d)                              # [K*N, d, 3d]
        qkv_h = torch.bmm(x_h, w)                                           # [K*N, BT, 3d]
        qkv = qkv_h.reshape(K, N, B, T, 3 * d).permute(2, 3, 0, 1, 4)       # [B,T,K,N,3d] head order
        q_head, k_head, v_head = qkv.split(d, dim=-1)                       # [B,T,K,N,d] head order

        # Route HEAD->STREAM: for each stream position s choose head p[k,s]
        p_idx = p.view(1, 1, K, N, 1).expand(B, T, K, N, d)
        q_stream = torch.gather(q_head, dim=3, index=p_idx)  # [B,T,K,N,d] (dim=3 now stream)
        k_stream = torch.gather(k_head, dim=3, index=p_idx)
        v_stream = torch.gather(v_head, dim=3, index=p_idx)

        # Pack to SDPA: [B,T,K,N,d] -> [B,N,K,T,d] -> [B,N*K,T,d]
        q = q_stream.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
        k = k_stream.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
        v = v_stream.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
        return q, k, v

    def _project_qkv_big_per_scale(self, x_in: torch.Tensor, phase: int):
        """
        STREAM/SCALES path:
        x_in: [B,T,N,in_dim] (LN already applied)

        Returns:
        q,k,v: [B, N*K, T, d]

        Strategy:
        - PRIVATE: vectorize across K with a single einsum (no permutations needed, no Kx input replication)
        - SHARED : keep per-scale loop to avoid materializing [B,T,K,N,in_dim] (which is huge for STREAM)
        """
        B, T, N, in_dim = x_in.shape
        assert N == self.N and in_dim == self.head_input_dim

        d = self.d
        K = self.K

        if self.stream_head_assignment == StreamHeadAssignment.PRIVATE:
            # Vectorized: qkv [B,T,K,N,3d]
            # No permutations, so this avoids the K-loop without replicating x_in.
            qkv = torch.einsum("b t n i, k n i o -> b t k n o", x_in, self.Wqkv)  # [B,T,K,N,3d]
            q, k, v = qkv.split(d, dim=-1)  # each [B,T,K,N,d]

            # Pack to SDPA format: [B,T,K,N,d] -> [B,N,K,T,d] -> [B,N*K,T,d]
            q = q.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            k = k.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            v = v.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            return q, k, v

        # SHARED: keep the loop to avoid Kx input materialization.
        q_list, k_list, v_list = [], [], []
        for k_idx in range(K):
            inv = self.invperm_within_scale[phase, k_idx]  # [N] head->stream
            x_head = x_in.index_select(dim=2, index=inv)   # [B,T,N,in_dim] head order

            # bmm across heads=N
            x_h = x_head.permute(2, 0, 1, 3).reshape(N, B * T, in_dim)      # [N, BT, in]
            w = self.Wqkv[k_idx]                                            # [N, in, 3d]
            qkv_h = torch.bmm(x_h, w)                                       # [N, BT, 3d]
            qkv = qkv_h.reshape(N, B, T, 3 * d).permute(1, 2, 0, 3)         # [B,T,N,3d] head order
            q_head, k_head, v_head = qkv.split(d, dim=-1)                   # each [B,T,N,d] head order

            # Route back to stream order
            p = self.perm_within_scale[phase, k_idx]                        # [N] stream->head
            q_stream = q_head.index_select(dim=2, index=p)
            k_stream = k_head.index_select(dim=2, index=p)
            v_stream = v_head.index_select(dim=2, index=p)

            q_list.append(q_stream.permute(0, 2, 1, 3))  # [B,N,T,d]
            k_list.append(k_stream.permute(0, 2, 1, 3))
            v_list.append(v_stream.permute(0, 2, 1, 3))

        q = torch.stack(q_list, dim=2).reshape(B, N * K, T, d)
        k = torch.stack(k_list, dim=2).reshape(B, N * K, T, d)
        v = torch.stack(v_list, dim=2).reshape(B, N * K, T, d)
        return q, k, v



    # -----------------------------
    # Core micro-step
    # -----------------------------
    def step(self, z: torch.Tensor, t: int, is_causal: bool) -> torch.Tensor:
        """
        z: [B,T,N,D]
        """
        B, T, N, D = z.shape
        assert N == self.N and D == self.D

        phase = t % self.N
        active_slots = self.active_slots[phase]  # [K]

        # Mixer
        z_head = z + self.dropout_mixer(self.mixer(self.ln_mixer(z)))  # [B,T,N,D]

        # Slot view
        z_slots = z_head.view(B, T, N, self.S, self.d)                 # view
        z_active = z_slots.index_select(3, active_slots)               # [B,T,N,K,d]

        # Build Q/K/V
        if self.head_input_scope == HeadInputScope.SLOT:
            # LN over last dim d for each active slot
            x_slot = self.ln_attn(z_active)  # [B,T,N,K,d]
            q, k, v = self._project_qkv_slot_vectorized(x_slot, phase)

        elif self.head_input_scope == HeadInputScope.SCALES:
            # LN over concatenated active slots
            x_scales = self.ln_attn(z_active.reshape(B, T, N, self.K * self.d))  # [B,T,N,K*d]
            q, k, v = self._project_qkv_big_per_scale(x_scales, phase)

        else:  # STREAM
            x_stream = self.ln_attn(z_head)  # [B,T,N,D]
            q, k, v = self._project_qkv_big_per_scale(x_stream, phase)

        # SDPA
        drop_p = self.dropout_p if (self.training and torch.is_grad_enabled()) else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=drop_p)  # [B, N*K, T, d]
        out = self.dropout_attn(out)

        # Back to [B,T,N,K,d]
        out = out.reshape(B, N, self.K, T, self.d).permute(0, 3, 1, 2, 4).contiguous()
        out = out + z_active

        # Write-back only active slots: clone flat state once
        z_next = z_head.clone()
        z_next_slots = z_next.view(B, T, N, self.S, self.d)
        z_next_slots.index_copy_(3, active_slots, out)

        return z_next

    # -----------------------------
    # Cycle and forward (burn-in + graft + short unroll)
    # -----------------------------
    def forward_cycle(self, z: torch.Tensor, t0: int, is_causal: bool) -> torch.Tensor:
        for dt in range(self.N):
            z = self.step(z, t=t0 + dt, is_causal=is_causal)
        return z

    def forward(self, x: torch.Tensor, R: int, is_causal: bool, grad_cycles: int = 1) -> torch.Tensor:
        """
        R: total cycles
        grad_cycles: number of final cycles to backprop through (1 = last cycle only)
        """
        assert 1 <= grad_cycles <= R
        burn_cycles = R - grad_cycles

        # encoder must have grad so stacking works
        z0 = self.encoder(x)   # [B,T,N,D] with grad

        # burn-in without graph
        z = z0.detach()
        t = 0
        if burn_cycles > 0:
            with torch.inference_mode():
                for _ in range(burn_cycles):
                    z = self.forward_cycle(z, t0=t, is_causal=is_causal)
                    t += self.N

        # gradient graft: value from burned-in z, gradient from z0
        z = z.detach() + (z0 - z0.detach())

        # final cycles with grad
        for _ in range(grad_cycles):
            z = self.forward_cycle(z, t0=t, is_causal=is_causal)
            t += self.N

        return self.decoder(z)


# -----------------------------
# Example / sanity test
# -----------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    B, T, io_dim = 4, 10, 16
    time_scales = [1, 2, 4]
    d = 12
    mixer_dim = 72
    enc_dec_rank = 8

    model = Morpher(
        io_dim=io_dim,
        d=d,
        time_scales=time_scales,
        enc_dec_rank=enc_dec_rank,
        mixer_hidden_dim=mixer_dim,
        stream_head_assignment=StreamHeadAssignment.PRIVATE,
        head_input_scope=HeadInputScope.SLOT,  # try STREAM / SCALES too
        dropout=0.1,
    )

    x = torch.randn(B, T, io_dim)

    # If on GPU/H100, wrap in autocast(bf16) for best perf
    # device = "cuda"
    # model = model.to(device)
    # x = x.to(device)
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #     y = model(x, R=4, is_causal=True, grad_cycles=1)

    y = model(x, R=2, is_causal=True, grad_cycles=1)
    print("x:", x.shape, "y:", y.shape)
