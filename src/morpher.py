from __future__ import annotations

from typing import List, Tuple
from enum import StrEnum
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NoParamRMSNorm, CastedLinear, SwiGLU, trunc_normal_init_, Dropout, lcm_list


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

        self.enc_base = CastedLinear(input_dim, output_dim, bias=False)
        self.enc_A = CastedLinear(input_dim, rank, bias=False)

        # [N, r, D] is nicer for bmm
        self.enc_B = nn.Parameter(
            torch.zeros(num_splits, rank, output_dim)
        )  # [N, r, D]

        # start near-off but not exactly 0 (for gradient flow)
        self.enc_beta = nn.Parameter(torch.tensor(0.01))

        self.reset_parameters()

    def reset_parameters(self):
        self.enc_base.reset_parameters()  # CastedLinear uses trunc_normal_init_
        self.enc_A.reset_parameters()     # CastedLinear uses trunc_normal_init_
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
        h = self.enc_A(x)  # [B,T,r]

        # Compute delta via batched GEMM:
        #   For each stream n: delta[:,:,n,:] = h @ B_n
        # Reshape h to [B*T, r], then expand to [N, B*T, r] without copying
        h_bt = h.reshape(B * T, self.rank)  # [BT, r]
        h_n = h_bt.unsqueeze(0).expand(self.N, B * T, self.rank)  # [N, BT, r]

        # enc_B: [N, r, D]
        delta_n = torch.bmm(h_n, self.enc_B)  # [N, BT, D]
        delta = delta_n.permute(1, 0, 2).reshape(
            B, T, self.N, self.output_dim
        )  # [B,T,N,D]

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
        self.ln = NoParamRMSNorm(in_dim) if use_layernorm else nn.Identity()
        self.proj_down = CastedLinear(in_dim, rank, bias=bias)
        self.proj_up = CastedLinear(rank, output_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.proj_down.reset_parameters()  # CastedLinear uses trunc_normal_init_
        self.proj_up.reset_parameters()    # CastedLinear uses trunc_normal_init_

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
    SHARED = "shared"  # phase-permuted within scale
    PRIVATE = "private"  # per-stream head within scale (identity)


class HeadInputScope(StrEnum):
    SLOT = "slot"
    STREAM = "stream"
    SCALES = "scales"



# =============================================================================
# Phase tables (init-time only)
# =============================================================================
class PhaseTables(nn.Module):
    """
    Owns all precomputed tables:
      active_slots[phase, k] -> which slot index is active for scale k at phase
      perm[phase, k, stream] -> head id used by that stream (stream->head)
      invperm[phase, k, head] -> stream that should feed that head (head->stream)
    """
    active_slots: torch.Tensor
    perm: torch.Tensor
    invperm: torch.Tensor

    def __init__(self, time_scales: List[int], assignment: StreamHeadAssignment):
        super().__init__()
        time_scales = sorted(time_scales)
        assert time_scales[0] == 1, "include scale=1 as fastest"

        self.time_scales = time_scales
        self.K = len(time_scales)
        self.N = lcm_list(time_scales)

        self.register_buffer("active_slots", self._build_active_slots(), persistent=False)
        perm, inv = self._build_perms(assignment)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("invperm", inv, persistent=False)

    def _build_active_slots(self) -> torch.Tensor:
        active = torch.empty(self.N, self.K, dtype=torch.long)
        for phase in range(self.N):
            off = 0
            for k, s in enumerate(self.time_scales):
                active[phase, k] = off + (phase % s)
                off += s
        return active  # [N, K]

    def _build_perms(self, assignment: StreamHeadAssignment) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = torch.empty(self.N, self.K, self.N, dtype=torch.long)
        inv  = torch.empty(self.N, self.K, self.N, dtype=torch.long)
        stream_ids = torch.arange(self.N, dtype=torch.long)

        for phase in range(self.N):
            for k, s in enumerate(self.time_scales):
                if assignment == StreamHeadAssignment.PRIVATE:
                    p = stream_ids
                else:
                    alpha = self.N // s
                    j = phase % s
                    p = (stream_ids + alpha * j) % self.N

                perm[phase, k] = p
                inv_p = torch.empty_like(p)
                inv_p[p] = stream_ids
                inv[phase, k] = inv_p

        return perm, inv  # each [N, K, N]


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
        mixer_expansion: float = 4.0,
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
        self.dropout_mixer = (
            Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )
        self.dropout_attn = (
            Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

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
        self.mixer_expansion = mixer_expansion
        self.ln_mixer = NoParamRMSNorm(self.D)
        self.mixer = SwiGLU(self.D, self.mixer_expansion)

        # Attention input dims per scope
        if self.head_input_scope == HeadInputScope.SLOT:
            self.head_input_dim = self.d
        elif self.head_input_scope == HeadInputScope.SCALES:
            self.head_input_dim = self.K * self.d
        else:
            self.head_input_dim = self.D

        self.ln_attn = NoParamRMSNorm(self.head_input_dim)

        # Fused Wqkv: [K, N, in_dim, 3d]
        self.Wqkv = nn.Parameter(
            torch.empty(self.K, self.N, self.head_input_dim, 3 * self.d)
        )

        # If SHARED, vectorizing across K requires materializing x_head = [B,T,K,N,in_dim].
        # We only do that if it stays under this budget.
        self.max_shared_xhead_bytes = (
            128 * 1024 * 1024
        )  # 128MB default (safe for most cases)

        self.phase_tables = PhaseTables(self.time_scales, self.stream_head_assignment)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize Wqkv with truncated normal (matches HRM's approach)
        wqkv_flat = self.Wqkv.view(-1, self.head_input_dim, 3 * self.d)
        trunc_normal_init_(wqkv_flat, std=1.0 / (self.head_input_dim ** 0.5))
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.mixer.reset_parameters()

    # -----------------------------
    # Tables
    # -----------------------------

    # -----------------------------
    # QKV projection helpers
    # -----------------------------
    def _project_qkv_slot(
        self, x_head: torch.Tensor, B: int, T: int, N: int, K: int, d: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Common QKV projection logic for both PRIVATE and SHARED wiring.
        x_head: [B,T,K,N,d] in head order
        Returns: q_head, k_head, v_head: [B,T,K,N,d] in head order
        """

        assert self.head_input_dim == d, (
            f"Common SLOT projector assumes in_dim == d, got {self.head_input_dim} != {d}"
        )

        # Pack (K*N) heads as batch for bmm
        x_h = x_head.permute(2, 3, 0, 1, 4).reshape(K * N, B * T, d)  # [K*N, BT, d]
        w = self.Wqkv.reshape(K * N, d, 3 * d)  # [K*N, d, 3d]
        qkv_h = torch.bmm(x_h, w)  # [K*N, BT, 3d]
        qkv = qkv_h.reshape(K, N, B, T, 3 * d).permute(
            2, 3, 0, 1, 4
        )  # [B,T,K,N,3d] head order
        q_head, k_head, v_head = qkv.split(d, dim=-1)  # [B,T,K,N,d] head order
        return q_head, k_head, v_head

    def _pack_qkv_for_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
        N: int,
        K: int,
        d: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Expect either [B,T,K,N,d] OR [B,N,K,T,d]
        if q.shape == (B, T, K, N, d):
            q_out = q.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            k_out = k.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            v_out = v.permute(0, 3, 2, 1, 4).reshape(B, N * K, T, d)
            return q_out, k_out, v_out

        if q.shape == (B, N, K, T, d):
            # This is already [B,N,K,T,d] so just reshape
            q_out = q.reshape(B, N * K, T, d)
            k_out = k.reshape(B, N * K, T, d)
            v_out = v.reshape(B, N * K, T, d)
            return q_out, k_out, v_out

        raise ValueError(
            f"Unexpected q shape {tuple(q.shape)} (expected {(B, T, K, N, d)} or {(B, N, K, T, d)})"
        )

    def _project_qkv_slot_vectorized(
        self, x_slot: torch.Tensor, phase: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            q_head, k_head, v_head = self._project_qkv_slot(x_head, B, T, N, K, d)
            return self._pack_qkv_for_sdpa(q_head, k_head, v_head, B, T, N, K, d)

        # SHARED wiring: permute streams -> heads and back
        p: torch.Tensor = self.phase_tables.perm[phase]  # [K,N] stream->head
        inv: torch.Tensor = self.phase_tables.invperm[phase]  # [K,N] head->stream

        # gather streams into HEAD order along dim=3 (stream axis)
        inv_idx = inv.view(1, 1, K, N, 1).expand(B, T, K, N, d)
        x_head = torch.gather(x_in, dim=3, index=inv_idx)  # [B,T,K,N,d] head order

        # Project all (K*N) heads in one batched GEMM
        q_head, k_head, v_head = self._project_qkv_slot(x_head, B, T, N, K, d)

        # Route HEAD->STREAM: for each stream position s choose head p[k,s]
        p_idx = p.view(1, 1, K, N, 1).expand(B, T, K, N, d)
        q_stream = torch.gather(
            q_head, dim=3, index=p_idx
        )  # [B,T,K,N,d] (dim=3 now stream)
        k_stream = torch.gather(k_head, dim=3, index=p_idx)
        v_stream = torch.gather(v_head, dim=3, index=p_idx)

        # Pack to SDPA format
        return self._pack_qkv_for_sdpa(q_stream, k_stream, v_stream, B, T, N, K, d)

    def _project_qkv_big_per_scale(self, x_in: torch.Tensor, phase: int):
        """
        STREAM/SCALES path:
        x_in: [B,T,N,in_dim] (LN already applied)
        Returns:
        q,k,v: [B, N*K, T, d]

        Strategy:
        - PRIVATE: vectorize across K via einsum (no Kx input materialization)
        - SHARED:
            * if Kx materialization is small enough: vectorize across K with one batched bmm
            * else: per-scale loop (memory-safe, best for large STREAM dims)
        """
        B, T, N, in_dim = x_in.shape
        assert N == self.N and in_dim == self.head_input_dim
        d = self.d
        K = self.K

        # ---------- PRIVATE: fully vectorized, no Kx input materialization ----------
        if self.stream_head_assignment == StreamHeadAssignment.PRIVATE:
            # qkv: [B,T,K,N,3d]
            qkv = torch.einsum("b t n i, k n i o -> b t k n o", x_in, self.Wqkv)
            q, k, v = qkv.split(d, dim=-1)  # each [B,T,K,N,d]
            return self._pack_qkv_for_sdpa(q, k, v, B, T, N, K, d)

        # ---------- SHARED: auto-select vectorized vs loop ----------
        # Vectorized SHARED needs x_head of shape [B,T,K,N,in_dim] (head order),
        # which is x_in.numel() * K elements.
        bytes_x_head = x_in.numel() * K * x_in.element_size()
        can_vectorize = bytes_x_head <= self.max_shared_xhead_bytes

        if can_vectorize:
            # Build x_broadcast: [B,T,K,N,in_dim] (stream order) as a view (expand is cheap)
            x_b = x_in.unsqueeze(2).expand(B, T, K, N, in_dim)

            inv: torch.Tensor = self.phase_tables.invperm[phase]  # [K,N] head->stream
            inv_idx = inv.view(1, 1, K, N, 1).expand(B, T, K, N, in_dim)

            # Reorder streams -> head order for all scales at once
            x_head = torch.gather(
                x_b, dim=3, index=inv_idx
            )  # [B,T,K,N,in_dim] head order

            # Project all K*N heads in one batched bmm
            x_h = x_head.permute(2, 3, 0, 1, 4).reshape(
                K * N, B * T, in_dim
            )  # [K*N, BT, in]
            w = self.Wqkv.reshape(K * N, in_dim, 3 * d)  # [K*N, in, 3d]
            qkv_h = torch.bmm(x_h, w)  # [K*N, BT, 3d]
            qkv = qkv_h.reshape(K, N, B, T, 3 * d).permute(
                2, 3, 0, 1, 4
            )  # [B,T,K,N,3d] head order
            q_head, k_head, v_head = qkv.split(d, dim=-1)  # [B,T,K,N,d] head order

            # Route head -> stream: for each stream position s choose head p[k,s]
            p: torch.Tensor = self.phase_tables.perm[phase]  # [K,N] stream->head
            p_idx = p.view(1, 1, K, N, 1).expand(B, T, K, N, d)

            q_stream = torch.gather(
                q_head, dim=3, index=p_idx
            )  # [B,T,K,N,d] (dim=3 now stream)
            k_stream = torch.gather(k_head, dim=3, index=p_idx)
            v_stream = torch.gather(v_head, dim=3, index=p_idx)

            return self._pack_qkv_for_sdpa(q_stream, k_stream, v_stream, B, T, N, K, d)

        # ---------- SHARED fallback: per-scale loop (memory-safe) ----------
        q_list, k_list, v_list = [], [], []
        for k_idx in range(K):
            inv_k: torch.Tensor = self.phase_tables.invperm[phase, k_idx]  # [N] head->stream
            x_head_k = x_in.index_select(
                dim=2, index=inv_k
            )  # [B,T,N,in_dim] head order

            # bmm across heads=N
            x_h = x_head_k.permute(2, 0, 1, 3).reshape(N, B * T, in_dim)  # [N, BT, in]
            w_k = self.Wqkv[k_idx]  # [N, in, 3d]
            qkv_h = torch.bmm(x_h, w_k)  # [N, BT, 3d]
            qkv = qkv_h.reshape(N, B, T, 3 * d).permute(
                1, 2, 0, 3
            )  # [B,T,N,3d] head order
            q_head, k_head, v_head = qkv.split(d, dim=-1)  # [B,T,N,d] head order

            # Route back to stream order
            p_k: torch.Tensor = self.phase_tables.perm[phase, k_idx]  # [N] stream->head
            q_stream = q_head.index_select(dim=2, index=p_k)
            k_stream = k_head.index_select(dim=2, index=p_k)
            v_stream = v_head.index_select(dim=2, index=p_k)

            q_list.append(q_stream.permute(0, 2, 1, 3))  # [B,N,T,d]
            k_list.append(k_stream.permute(0, 2, 1, 3))
            v_list.append(v_stream.permute(0, 2, 1, 3))

        q_stacked = torch.stack(q_list, dim=2)  # [B,N,K,T,d]
        k_stacked = torch.stack(k_list, dim=2)
        v_stacked = torch.stack(v_list, dim=2)
        return self._pack_qkv_for_sdpa(q_stacked, k_stacked, v_stacked, B, T, N, K, d)

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
        active_slots: torch.Tensor = self.phase_tables.active_slots[phase]  # [K]

        # Mixer
        z_head = z + self.dropout_mixer(self.mixer(self.ln_mixer(z)))  # [B,T,N,D]

        # Slot view
        z_slots = z_head.view(B, T, N, self.S, self.d)  # view
        z_active = z_slots.index_select(3, active_slots)  # [B,T,N,K,d]

        # Build Q/K/V
        if self.head_input_scope == HeadInputScope.SLOT:
            # LN over last dim d for each active slot
            x_slot = self.ln_attn(z_active)  # [B,T,N,K,d]
            q, k, v = self._project_qkv_slot_vectorized(x_slot, phase)

        elif self.head_input_scope == HeadInputScope.SCALES:
            # LN over concatenated active slots
            x_scales = self.ln_attn(
                z_active.reshape(B, T, N, self.K * self.d)
            )  # [B,T,N,K*d]
            q, k, v = self._project_qkv_big_per_scale(x_scales, phase)

        else:  # STREAM
            x_stream = self.ln_attn(z_head)  # [B,T,N,D]
            q, k, v = self._project_qkv_big_per_scale(x_stream, phase)

        # SDPA
        drop_p = self.dropout_p if (self.training and torch.is_grad_enabled()) else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=drop_p
        )  # [B, N*K, T, d]
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

    def forward(
        self, x: torch.Tensor, R: int, is_causal: bool, grad_cycles: int = 1
    ) -> torch.Tensor:
        """
        R: total cycles
        grad_cycles: number of final cycles to backprop through (1 = last cycle only)
        """
        assert 1 <= grad_cycles <= R
        burn_cycles = R - grad_cycles

        # encoder must have grad so stacking works
        z0 = self.encoder(x)  # [B,T,N,D] with grad

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
    mixer_expansion = 4.0  # Expansion factor for SwiGLU
    enc_dec_rank = 8

    model = Morpher(
        io_dim=io_dim,
        d=d,
        time_scales=time_scales,
        enc_dec_rank=enc_dec_rank,
        mixer_expansion=mixer_expansion,
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
