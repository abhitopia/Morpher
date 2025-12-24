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

class AttnBackend(StrEnum):
    SDPA = "sdpa"
    FLASH3 = "flash3"                    # flash_attn_func(q,k,v) with q,k,v [B,T,H,d]
    FLASH3_QKVPACKED = "flash3_qkvpacked" # flash_attn_qkvpacked_func(qkv) with qkv [B,T,3,H,d]

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


# =============================================================================
# QKV projection + routing (returns packed QKV once)
# =============================================================================
def _streams_to_heads(x_btkni: torch.Tensor, inv_kn: torch.Tensor) -> torch.Tensor:
    """
    x_btkni: [B,T,K,N,in] in stream-order along N
    inv_kn:  [K,N] head->stream
    returns: [B,T,K,N,in] in head-order along N
    """
    B, T, K, N, in_dim = x_btkni.shape
    idx = inv_kn.view(1, 1, K, N, 1).expand(B, T, K, N, in_dim)
    return torch.gather(x_btkni, dim=3, index=idx)


def _heads_to_streams(x_btkni: torch.Tensor, perm_kn: torch.Tensor) -> torch.Tensor:
    """
    x_btkni: [B,T,K,N,in] in head-order along N
    perm_kn: [K,N] stream->head
    returns: [B,T,K,N,in] in stream-order along N
    """
    B, T, K, N, in_dim = x_btkni.shape
    idx = perm_kn.view(1, 1, K, N, 1).expand(B, T, K, N, in_dim)
    return torch.gather(x_btkni, dim=3, index=idx)


def _project_bmm(x_btkni: torch.Tensor, w_knio: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T,K,N,in]
    w: [K,N,in,3d]
    -> qkv: [B,T,K,N,3d]
    """
    B, T, K, N, in_dim = x_btkni.shape
    x = x_btkni.permute(2, 3, 0, 1, 4).reshape(K * N, B * T, in_dim)     # [K*N, BT, in]
    w = w_knio.reshape(K * N, in_dim, -1)                                 # [K*N, in, 3d]
    y = torch.bmm(x, w).reshape(K, N, B, T, -1).permute(2, 3, 0, 1, 4)    # [B,T,K,N,3d]
    return y


class QKVProjector(nn.Module):
    """
    Owns Wqkv and produces packed QKV in *stream order*:
      qkv_stream: [B,T,K,N,3d]
    """
    def __init__(
        self,
        *,
        K: int,
        N: int,
        d: int,
        in_dim: int,
        assignment: StreamHeadAssignment,
        max_shared_xhead_bytes: int = 128 * 1024 * 1024,
    ):
        super().__init__()
        self.K = K
        self.N = N
        self.d = d
        self.in_dim = in_dim
        self.assignment = assignment
        self.max_shared_xhead_bytes = int(max_shared_xhead_bytes)

        self.Wqkv = nn.Parameter(torch.empty(K, N, in_dim, 3 * d))

    def reset_parameters(self) -> None:
        w_flat = self.Wqkv.view(-1, self.in_dim, 3 * self.d)
        trunc_normal_init_(w_flat, std=1.0 / (self.in_dim ** 0.5))

    def project_slot(
        self,
        x_btnkd: torch.Tensor,  # [B,T,N,K,d] stream-order
        *,
        phase: int,
        perm: torch.Tensor,     # [Nphase,K,N]
        invperm: torch.Tensor,  # [Nphase,K,N]
    ) -> torch.Tensor:
        """
        SLOT path (in_dim == d):
          x_btnkd: [B,T,N,K,d]
        returns:
          qkv_stream: [B,T,K,N,3d]
        """
        B, T, N, K, d = x_btnkd.shape
        assert (N, K, d) == (self.N, self.K, self.d)
        assert self.in_dim == d

        # -> [B,T,K,N,d]
        x_btknd = x_btnkd.permute(0, 1, 3, 2, 4)

        if self.assignment == StreamHeadAssignment.PRIVATE:
            # head-order == stream-order along N
            return _project_bmm(x_btknd, self.Wqkv)  # [B,T,K,N,3d]

        inv_kn = invperm[phase]                      # [K,N] head->stream
        x_head = _streams_to_heads(x_btknd, inv_kn)   # [B,T,K,N,d] head-order

        qkv_head = _project_bmm(x_head, self.Wqkv)     # [B,T,K,N,3d] head-order

        perm_kn = perm[phase]                         # [K,N] stream->head
        qkv_stream = _heads_to_streams(qkv_head, perm_kn)
        return qkv_stream

    def project_stream(
        self,
        x_btni: torch.Tensor,   # [B,T,N,in_dim] stream-order
        *,
        phase: int,
        perm: torch.Tensor,
        invperm: torch.Tensor,
    ) -> torch.Tensor:
        """
        STREAM / SCALES path:
          x_btni: [B,T,N,in_dim]
        returns:
          qkv_stream: [B,T,K,N,3d]
        """
        B, T, N, in_dim = x_btni.shape
        assert (N, in_dim) == (self.N, self.in_dim)
        K, d = self.K, self.d

        # PRIVATE: fully vectorized without materializing Kx input
        if self.assignment == StreamHeadAssignment.PRIVATE:
            # [B,T,N,in] x [K,N,in,3d] -> [B,T,K,N,3d]
            return torch.einsum("btni,knio->btkno", x_btni, self.Wqkv)

        # SHARED: choose vectorized vs loop (memory budget)
        bytes_x_head = x_btni.numel() * K * x_btni.element_size()
        can_vectorize = bytes_x_head <= self.max_shared_xhead_bytes

        if can_vectorize:
            # broadcast to [B,T,K,N,in]
            x_btkni = x_btni.unsqueeze(2).expand(B, T, K, N, in_dim)
            inv_kn = invperm[phase]                    # [K,N]
            x_head = _streams_to_heads(x_btkni, inv_kn) # [B,T,K,N,in] head-order

            qkv_head = _project_bmm(x_head, self.Wqkv)  # [B,T,K,N,3d] head-order

            perm_kn = perm[phase]                       # [K,N]
            return _heads_to_streams(qkv_head, perm_kn)  # [B,T,K,N,3d] stream-order

        # fallback loop over scales (memory-safe)
        qkv_per_k: List[torch.Tensor] = []
        for k_idx in range(K):
            inv_k = invperm[phase, k_idx]                  # [N] head->stream
            x_head_k = x_btni.index_select(dim=2, index=inv_k)  # [B,T,N,in] head-order

            # bmm across N heads
            x_h = x_head_k.permute(2, 0, 1, 3).reshape(N, B * T, in_dim)  # [N, BT, in]
            w_k = self.Wqkv[k_idx]                                        # [N, in, 3d]
            qkv_h = torch.bmm(x_h, w_k)                                   # [N, BT, 3d]
            qkv_head_k = qkv_h.reshape(N, B, T, 3 * d).permute(1, 2, 0, 3) # [B,T,N,3d] head-order

            perm_k = perm[phase, k_idx]                                   # [N] stream->head
            qkv_stream_k = qkv_head_k.index_select(dim=2, index=perm_k)   # [B,T,N,3d] stream-order
            qkv_per_k.append(qkv_stream_k.unsqueeze(2))                   # [B,T,1,N,3d]

        return torch.cat(qkv_per_k, dim=2)  # [B,T,K,N,3d]


# =============================================================================
# Attention adapter (SDPA vs FlashAttention-3) with one internal QKV format
# =============================================================================
class AttentionAdapter(nn.Module):
    """
    Internal QKV format:
      qkv_stream: [B,T,K,N,3d] (stream-order along N)
    Output:
      out: [B,T,N,K,d] (stream-order along N, matches z_active)
    """
    def __init__(self, backend: AttnBackend):
        super().__init__()
        self.backend = backend

        self._flash_attn_func = None
        self._flash_qkvpacked_func = None
        if backend in (AttnBackend.FLASH3, AttnBackend.FLASH3_QKVPACKED):
            try:
                from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
            except Exception as e:
                raise ImportError(
                    "Requested FlashAttention-3 backend but flash_attn is not importable. "
                    "Install flash-attn (and ensure you're on a supported CUDA setup), "
                    "or choose AttnBackend.SDPA."
                ) from e
            self._flash_attn_func = flash_attn_func
            self._flash_qkvpacked_func = flash_attn_qkvpacked_func

    @staticmethod
    def _effective_dropout_p(p: float, training: bool) -> float:
        return p if (training and torch.is_grad_enabled()) else 0.0

    def forward(
        self,
        qkv_stream_btk_n3d: torch.Tensor,  # [B,T,K,N,3d]
        *,
        is_causal: bool,
        dropout_p: float,
    ) -> torch.Tensor:
        B, T, K, N, three_d = qkv_stream_btk_n3d.shape
        assert three_d % 3 == 0
        d = three_d // 3
        H = N * K

        p = self._effective_dropout_p(dropout_p, self.training)

        # Pack heads as H = N*K in (N,K) order
        # [B,T,K,N,3d] -> [B,T,N,K,3d] -> [B,T,H,3d]
        qkv_btnk3d = qkv_stream_btk_n3d.permute(0, 1, 3, 2, 4).reshape(B, T, H, 3 * d)

        if self.backend == AttnBackend.SDPA:
            # SDPA: q,k,v [B,H,T,d]
            qkv = qkv_btnk3d.reshape(B, T, H, 3, d)
            q = qkv[..., 0, :].transpose(1, 2)  # [B,H,T,d]
            k = qkv[..., 1, :].transpose(1, 2)
            v = qkv[..., 2, :].transpose(1, 2)

            out_bhtd = F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, dropout_p=p
            )  # [B,H,T,d]
            out_bthd = out_bhtd.transpose(1, 2)  # [B,T,H,d]

        elif self.backend == AttnBackend.FLASH3:
            # FlashAttn: q,k,v [B,T,H,d]
            qkv = qkv_btnk3d.reshape(B, T, H, 3, d)
            q = qkv[..., 0, :].contiguous()
            k = qkv[..., 1, :].contiguous()
            v = qkv[..., 2, :].contiguous()
            out_bthd = self._flash_attn_func(q, k, v, dropout_p=p, causal=is_causal)

        else:  # FLASH3_QKVPACKED
            # qkvpacked: [B,T,3,H,d]
            qkv = qkv_btnk3d.reshape(B, T, H, 3, d).permute(0, 1, 3, 2, 4).contiguous()
            out_bthd = self._flash_qkvpacked_func(qkv, dropout_p=p, causal=is_causal)

        # Unpack back to [B,T,N,K,d]
        out_btnkd = out_bthd.reshape(B, T, N, K, d)
        return out_btnkd

# -----------------------------
# Morpher
# -----------------------------
class Morpher(nn.Module):

    def __init__(
        self,
        *,
        io_dim: int,
        d: int,
        time_scales: List[int],
        enc_dec_rank: int,
        mixer_expansion: float = 4.0,
        stream_head_assignment: StreamHeadAssignment = StreamHeadAssignment.SHARED,
        head_input_scope: HeadInputScope = HeadInputScope.SLOT,
        attn_backend: AttnBackend = AttnBackend.SDPA,
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

        # Tables (init-time)
        self.tables = PhaseTables(self.time_scales, self.stream_head_assignment)


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


        # QKV projector produces packed [B,T,K,N,3d]
        self.projector = QKVProjector(
            K=self.K,
            N=self.N,
            d=self.d,
            in_dim=self.head_input_dim,
            assignment=self.stream_head_assignment,
            # If SHARED, vectorizing across K requires materializing x_head = [B,T,K,N,in_dim].
            # We only do that if it stays under this budget.
            max_shared_xhead_bytes=128 * 1024 * 1024,
        )

        # Attention backend adapter
        self.attn = AttentionAdapter(attn_backend)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.mixer.reset_parameters()
        self.projector.reset_parameters()


    # -----------------------------
    # Core micro-step
    # -----------------------------
    def step(self, z: torch.Tensor, t: int, is_causal: bool) -> torch.Tensor:
        """
        z: [B,T,N,D]
        """
        B, T, N, D = z.shape
        assert (N, D) == (self.N, self.D)

        phase = t % self.N
        active_slots: torch.Tensor = self.tables.active_slots[phase]  # [K]

        # Mixer
        z_head = z + self.dropout_mixer(self.mixer(self.ln_mixer(z)))  # [B,T,N,D]

        # Slot view
        z_slots = z_head.view(B, T, N, self.S, self.d)  # view
        z_active = z_slots.index_select(3, active_slots)  # [B,T,N,K,d]

        # Build Q/K/V
        if self.head_input_scope == HeadInputScope.SLOT:
            # LN over last dim d for each active slot
            x_slot = self.ln_attn(z_active)  # [B,T,N,K,d]
            qkv_stream = self.projector.project_slot(
                            x_slot,
                            phase=phase,
                            perm=self.tables.perm,
                            invperm=self.tables.invperm,
                        )
        elif self.head_input_scope == HeadInputScope.SCALES:
            # LN over concatenated active slots
            x_scales = self.ln_attn(z_active.reshape(B, T, N, self.K * self.d))  # [B,T,N,K*d]
            qkv_stream = self.projector.project_stream(
                x_scales,
                phase=phase,
                perm=self.tables.perm,
                invperm=self.tables.invperm,
            )
        else:  # STREAM
            x_stream = self.ln_attn(z_head)  # [B,T,N,D]
            qkv_stream = self.projector.project_stream(
                    x_stream,
                    phase=phase,
                    perm=self.tables.perm,
                    invperm=self.tables.invperm,
                )


        # Attend: out [B,T,N,K,d] in stream order
        out = self.attn(qkv_stream, is_causal=is_causal, dropout_p=self.dropout_p)
        out = self.dropout_attn(out)

        # Residual on active slots
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
