from __future__ import annotations

from enum import StrEnum
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    NoParamRMSNorm, CastedLinear, SwiGLU, trunc_normal_init_, Dropout, lcm_list,
    RotaryEmbedding, apply_rotary_pos_emb,
)


"""
================================================================================
Morpher (refactored)
================================================================================

High-level idea
---------------
We maintain N parallel "streams" of state. Each stream contains S "slots" and each
slot has width d, so the per-stream state width is:
    D = S * d

At each micro-step t we:
  1) Mix each stream's full state (per-stream MLP).
  2) Select K active slots per stream, where K = len(time_scales) and each scale
     activates exactly one slot at that micro-step.
  3) Build QKV for attention over streams (and scales), run attention, then write
     the attention outputs back only into the active slots.

Shape conventions
-----------------
We use a suffix naming convention to make tensor layouts obvious. Letters:
  b: batch (B)
  t: time  (T)
  n: stream id (N = lcm(time_scales))
  k: scale index (K = len(time_scales))
  s: slot index (S = sum(time_scales))
  d: per-slot dim (d)
  D: per-stream dim (D = S*d)
  i: generic projection input dim
  o: generic projection output dim

Core state:
  z_btnd : [B, T, N, D]

Slot view:
  z_btnsd: [B, T, N, S, d]

Active slots (selected from slot view):
  z_active_btnkd: [B, T, N, K, d]

Attention/QKV pipeline (chosen to be consistent at boundaries):
--------------------------------------------------------------
Inside the attention pipeline we use SCALE-MAJOR layout:
  ..._btknd : [B, T, K, N, d]     (N is still "stream order" for outputs/heads)
  qkv_btk_n3d: [B, T, K, N, 3d]

This makes projector <-> attention adapter boundaries consistent:
  projector(...) -> qkv_btk_n3d
  attn(qkv_btk_n3d) -> out_btknd
"""


# =============================================================================
# Stream LoRA Encoder / Decoder
# =============================================================================
class StreamLoRAEncoder(nn.Module):
    """
    Encode input x into an initial multi-stream state z0.

    Input:
      x_bti : [B, T, input_dim]

    Output:
      z0_btnd : [B, T, N, output_dim]

    Structure:
      base(x) is shared across streams
      delta(x) is stream-specific LoRA (A shared, B per-stream)
    """

    def __init__(self, input_dim: int, output_dim: int, num_splits: int, rank: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = num_splits
        self.rank = rank

        self.enc_base = CastedLinear(input_dim, output_dim, bias=False)  # shared
        self.enc_A = CastedLinear(input_dim, rank, bias=False)          # shared

        # Per-stream LoRA B: [N, r, D]
        self.enc_B = nn.Parameter(torch.zeros(num_splits, rank, output_dim))

        # A small scalar so delta starts "near off" but not exactly 0 for gradient flow
        self.enc_beta = nn.Parameter(torch.tensor(0.01))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.enc_base.reset_parameters()  # CastedLinear uses trunc_normal_init_
        self.enc_A.reset_parameters()     # CastedLinear uses trunc_normal_init_
        nn.init.zeros_(self.enc_B)
        with torch.no_grad():
            self.enc_beta.fill_(0.01)

    def forward(self, x_bti: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_bti.shape

        base_btd = self.enc_base(x_bti)   # [B, T, D]
        h_btr = self.enc_A(x_bti)         # [B, T, r]

        # Compute delta for each stream via batched GEMM:
        #   delta[:,:,n,:] = h @ B[n]
        #
        # Efficient grouping: bmm over N "batches"
        h_bt_r = h_btr.reshape(B * T, self.rank)                        # [BT, r]
        h_n_bt_r = h_bt_r.unsqueeze(0).expand(self.N, B * T, self.rank) # [N, BT, r]
        delta_n_bt_d = torch.bmm(h_n_bt_r, self.enc_B)                  # [N, BT, D]

        delta_btnd = delta_n_bt_d.permute(1, 0, 2).reshape(B, T, self.N, self.output_dim)
        return base_btd.unsqueeze(2) + self.enc_beta * delta_btnd       # [B, T, N, D]


class StreamLoRADecoder(nn.Module):
    """
    Decode multi-stream state back to output.

    Input:
      z_btnd : [B, T, N, input_dim]

    Output:
      y_bto : [B, T, output_dim]
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

        flat_in = self.N * self.input_dim
        self.ln = NoParamRMSNorm(flat_in) if use_layernorm else nn.Identity()
        self.proj_down = CastedLinear(flat_in, rank, bias=bias)
        self.proj_up = CastedLinear(rank, output_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.proj_down.reset_parameters()
        self.proj_up.reset_parameters()

    def forward(self, z_btnd: torch.Tensor) -> torch.Tensor:
        B, T, N, D = z_btnd.shape
        assert N == self.N and D == self.input_dim

        z_flat_bt_i = z_btnd.reshape(B, T, N * D)        # [B, T, N*D]
        h_btr = self.proj_down(self.ln(z_flat_bt_i))     # [B, T, r]
        return self.proj_up(h_btr)                       # [B, T, out_dim]


# =============================================================================
# Enums
# =============================================================================
class StreamHeadAssignment(StrEnum):
    SHARED = "shared"   # phase-dependent stream<->head permutation within each scale
    PRIVATE = "private" # stream == head (identity mapping)


class HeadInputScope(StrEnum):
    SLOT = "slot"       # each scale uses only its own active slot (dim=d)
    STREAM = "stream"   # each scale uses full stream state (dim=D)
    SCALES = "scales"   # each scale uses concat of active slots (dim=K*d)


class AttnBackend(StrEnum):
    SDPA = "sdpa"
    FLASH3 = "flash3"                     # flash_attn_func(q,k,v) with q,k,v [B,T,H,d]
    FLASH3_QKVPACKED = "flash3_qkvpacked" # flash_attn_qkvpacked_func(qkv) with qkv [B,T,3,H,d]


# =============================================================================
# Phase tables (init-time only)
# =============================================================================
class PhaseTables(nn.Module):
    """
    Precomputed routing tables for all phases.

    Let:
      time_scales = [s0, s1, ..., s_{K-1}]
      K = len(time_scales)
      N = lcm(time_scales)

    active_slots[phase, k] gives the slot index (0..S-1) that is active for scale k
    at that phase.

    stream_to_head[phase, k, stream] -> head
    head_to_stream[phase, k, head]   -> stream

    Notes:
      - In PRIVATE assignment: stream_to_head is identity for all phases/scales.
      - In SHARED assignment: stream_to_head is a phase-dependent cyclic shift that
        depends on the scale period.
    """

    active_slots: torch.Tensor
    stream_to_head: torch.Tensor
    head_to_stream: torch.Tensor

    def __init__(self, time_scales: List[int], assignment: StreamHeadAssignment):
        super().__init__()
        time_scales = sorted(time_scales)
        assert time_scales[0] == 1, "include scale=1 as fastest"

        self.time_scales = time_scales
        self.K = len(time_scales)
        self.N = lcm_list(time_scales)

        self.register_buffer("active_slots", self._build_active_slots(), persistent=False)
        s2h, h2s = self._build_stream_head_maps(assignment)
        self.register_buffer("stream_to_head", s2h, persistent=False)
        self.register_buffer("head_to_stream", h2s, persistent=False)

    def _build_active_slots(self) -> torch.Tensor:
        # Layout: [phase, k] -> slot index in [0, S)
        active_pk = torch.empty(self.N, self.K, dtype=torch.long)

        for phase in range(self.N):
            offset = 0
            for k, s in enumerate(self.time_scales):
                active_pk[phase, k] = offset + (phase % s)
                offset += s

        return active_pk  # [Nphase, K]

    def _build_stream_head_maps(
        self, assignment: StreamHeadAssignment
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # stream_to_head: [phase, k, stream] -> head
        # head_to_stream: [phase, k, head] -> stream
        s2h_pkn = torch.empty(self.N, self.K, self.N, dtype=torch.long)
        h2s_pkn = torch.empty(self.N, self.K, self.N, dtype=torch.long)
        stream_ids_n = torch.arange(self.N, dtype=torch.long)

        for phase in range(self.N):
            for k, s in enumerate(self.time_scales):
                if assignment == StreamHeadAssignment.PRIVATE:
                    s2h_n = stream_ids_n
                else:
                    # phase-dependent cyclic shift; alpha ensures period matches scale
                    alpha = self.N // s
                    j = phase % s
                    s2h_n = (stream_ids_n + alpha * j) % self.N  # [N]

                s2h_pkn[phase, k] = s2h_n

                h2s_n = torch.empty_like(s2h_n)
                h2s_n[s2h_n] = stream_ids_n
                h2s_pkn[phase, k] = h2s_n

        return s2h_pkn, h2s_pkn


# =============================================================================
# Routing + projection helpers (performance-critical)
# =============================================================================
def _route_streams_to_heads_btkni(x_stream_btkni: torch.Tensor, head_to_stream_kn: torch.Tensor) -> torch.Tensor:
    """
    Reorder N dimension from "stream order" -> "head order".

    x_stream_btkni: [B, T, K, N, in]
    head_to_stream_kn: [K, N] where head_to_stream_kn[k, head] = stream

    returns x_head_btkni: [B, T, K, N, in] (N is now head order)
    """
    B, T, K, N, in_dim = x_stream_btkni.shape
    idx = head_to_stream_kn.view(1, 1, K, N, 1).expand(B, T, K, N, in_dim)
    return torch.gather(x_stream_btkni, dim=3, index=idx)


def _route_heads_to_streams_btkni(x_head_btkni: torch.Tensor, stream_to_head_kn: torch.Tensor) -> torch.Tensor:
    """
    Reorder N dimension from "head order" -> "stream order".

    x_head_btkni: [B, T, K, N, in]
    stream_to_head_kn: [K, N] where stream_to_head_kn[k, stream] = head

    returns x_stream_btkni: [B, T, K, N, in] (N is now stream order)
    """
    B, T, K, N, in_dim = x_head_btkni.shape
    idx = stream_to_head_kn.view(1, 1, K, N, 1).expand(B, T, K, N, in_dim)
    return torch.gather(x_head_btkni, dim=3, index=idx)


def _project_bmm_btkni(x_btkni: torch.Tensor, w_knio: torch.Tensor) -> torch.Tensor:
    """
    Grouped projection implemented via batched bmm.

    x_btkni: [B, T, K, N, in]
    w_knio:  [K, N, in, out]   (out is typically 3*d)

    returns y_btkno: [B, T, K, N, out]
    """
    B, T, K, N, in_dim = x_btkni.shape
    x = x_btkni.permute(2, 3, 0, 1, 4).reshape(K * N, B * T, in_dim)  # [K*N, BT, in]
    w = w_knio.reshape(K * N, in_dim, -1)                              # [K*N, in, out]
    y = torch.bmm(x, w)                                                # [K*N, BT, out]
    return y.reshape(K, N, B, T, -1).permute(2, 3, 0, 1, 4)            # [B, T, K, N, out]


# =============================================================================
# QKV projector (returns QKV in a single packed tensor)
# =============================================================================
class QKVProjector(nn.Module):
    """
    Owns Wqkv and produces packed QKV in SCALE-MAJOR layout:

      qkv_btk_n3d: [B, T, K, N, 3d]   (N is stream-order on output)

    Internally, when StreamHeadAssignment.SHARED is used, we temporarily route
    streams -> heads before applying Wqkv, then route back.
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

        # [K, N, in_dim, 3d]
        self.Wqkv = nn.Parameter(torch.empty(K, N, in_dim, 3 * d))

    def reset_parameters(self) -> None:
        w_flat = self.Wqkv.view(-1, self.in_dim, 3 * self.d)
        trunc_normal_init_(w_flat, std=1.0 / (self.in_dim ** 0.5))

    def project_slot(
        self,
        x_btknd: torch.Tensor,  # [B, T, K, N, d] (stream-order along N)
        *,
        phase: int,
        stream_to_head: torch.Tensor,  # [Nphase, K, N]
        head_to_stream: torch.Tensor,  # [Nphase, K, N]
    ) -> torch.Tensor:
        """
        SLOT scope projection.

        Input:
          x_btknd : [B, T, K, N, d]
        Output:
          qkv_btk_n3d : [B, T, K, N, 3d]
        """
        B, T, K, N, d = x_btknd.shape
        assert (K, N, d) == (self.K, self.N, self.d)
        assert self.in_dim == d, "project_slot requires in_dim == d"

        if self.assignment == StreamHeadAssignment.PRIVATE:
            # No routing needed: head == stream
            return _project_bmm_btkni(x_btknd, self.Wqkv)

        # Route streams -> heads so head i gets the right stream's input at this phase.
        h2s_kn = head_to_stream[phase]                         # [K, N]
        x_head_btknd = _route_streams_to_heads_btkni(x_btknd, h2s_kn)

        # Apply per-(k,head) projection
        qkv_head_btk_n3d = _project_bmm_btkni(x_head_btknd, self.Wqkv)

        # Route heads -> streams so stream n receives the head assigned to it at this phase.
        s2h_kn = stream_to_head[phase]                         # [K, N]
        qkv_stream_btk_n3d = _route_heads_to_streams_btkni(qkv_head_btk_n3d, s2h_kn)
        return qkv_stream_btk_n3d

    def project_stream(
        self,
        x_btni: torch.Tensor,  # [B, T, N, in_dim] (stream-order)
        *,
        phase: int,
        stream_to_head: torch.Tensor,
        head_to_stream: torch.Tensor,
    ) -> torch.Tensor:
        """
        STREAM or SCALES scope projection.

        Input:
          x_btni : [B, T, N, in_dim]
        Output:
          qkv_btk_n3d : [B, T, K, N, 3d]
        """
        B, T, N, in_dim = x_btni.shape
        assert (N, in_dim) == (self.N, self.in_dim)

        # PRIVATE: fully vectorized without materializing K copies of x.
        if self.assignment == StreamHeadAssignment.PRIVATE:
            # x: [B,T,N,in] , W: [K,N,in,3d] -> [B,T,K,N,3d]
            return torch.einsum("btni,knio->btkno", x_btni, self.Wqkv)

        # SHARED: decide between vectorized broadcast vs loop based on memory budget.
        bytes_if_broadcast = x_btni.numel() * self.K * x_btni.element_size()
        can_vectorize = bytes_if_broadcast <= self.max_shared_xhead_bytes

        if can_vectorize:
            # Broadcast x across K: [B, T, 1, N, in] -> [B, T, K, N, in]
            x_btkni = x_btni.unsqueeze(2).expand(B, T, self.K, N, in_dim)

            # Route streams -> heads at this phase
            h2s_kn = head_to_stream[phase]  # [K, N]
            x_head_btkni = _route_streams_to_heads_btkni(x_btkni, h2s_kn)

            # Project
            qkv_head_btk_n3d = _project_bmm_btkni(x_head_btkni, self.Wqkv)

            # Route heads -> streams
            s2h_kn = stream_to_head[phase]  # [K, N]
            return _route_heads_to_streams_btkni(qkv_head_btk_n3d, s2h_kn)

        # Memory-safe fallback: loop over scales (K is typically small).
        qkv_per_k: List[torch.Tensor] = []
        for k in range(self.K):
            # head_to_stream for this scale: [N]
            h2s_n = head_to_stream[phase, k]  # [N]
            x_head_btni = x_btni.index_select(dim=2, index=h2s_n)  # [B, T, N, in] head-order

            # Grouped bmm across N heads for this one scale
            x_n_bti = x_head_btni.permute(2, 0, 1, 3).reshape(N, B * T, in_dim)  # [N, BT, in]
            w_nio = self.Wqkv[k]                                                 # [N, in, 3d]
            qkv_n_bto = torch.bmm(x_n_bti, w_nio)                                 # [N, BT, 3d]

            qkv_head_btn_3d = qkv_n_bto.reshape(N, B, T, 3 * self.d).permute(1, 2, 0, 3)  # [B, T, N, 3d]

            # route head-order -> stream-order
            s2h_n = stream_to_head[phase, k]  # [N]
            qkv_stream_btn_3d = qkv_head_btn_3d.index_select(dim=2, index=s2h_n)  # [B, T, N, 3d]
            qkv_per_k.append(qkv_stream_btn_3d.unsqueeze(2))                      # [B, T, 1, N, 3d]

        return torch.cat(qkv_per_k, dim=2)  # [B, T, K, N, 3d]


# =============================================================================
# Attention adapter (SDPA vs FlashAttention-3)
# =============================================================================
class AttentionAdapter(nn.Module):
    """
    Attention over (K * N) heads, where each (k, n) is a head.

    Input (consistent with QKVProjector):
      qkv_btk_n3d : [B, T, K, N, 3d]

    Output (consistent shape/layout):
      out_btknd : [B, T, K, N, d]

    We flatten heads as:
      H = K * N
      head_index = k * N + n    (scale-major head order)
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

    def _effective_dropout_p(self, p: float) -> float:
        # Match SDPA/FlashAttn conventions: dropout only when training *and* grads enabled.
        return p if (self.training and torch.is_grad_enabled()) else 0.0

    def forward(
        self,
        qkv_btk_n3d: torch.Tensor,  # [B, T, K, N, 3d]
        *,
        is_causal: bool,
        dropout_p: float,
        # RoPE (optional): both are [T, d], already sliced to the current sequence length
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, K, N, three_d = qkv_btk_n3d.shape
        assert three_d % 3 == 0
        d = three_d // 3
        H = K * N

        p = self._effective_dropout_p(dropout_p)

        # Flatten heads (k,n) -> h in scale-major order: h = k*N + n
        # [B, T, K, N, 3d] -> [B, T, H, 3d]
        qkv_bth_3d = qkv_btk_n3d.reshape(B, T, H, 3 * d)
        qkv_bth3d = qkv_bth_3d.reshape(B, T, H, 3, d)  # [B, T, H, 3, d]

        # --- Unpack q/k/v as [B, T, H, d] (shared across backends) ---
        q_bthd = qkv_bth3d[..., 0, :]
        k_bthd = qkv_bth3d[..., 1, :]
        v_bthd = qkv_bth3d[..., 2, :]

        # --- Apply RoPE to q/k if provided ---
        # apply_rotary_pos_emb expects:
        #   q,k: [B, T, H, d]
        #   cos,sin: [T, d]
        if cos is not None:
            # sin should be present too; keep a hard assert to catch wiring mistakes.
            assert sin is not None
            q_bthd, k_bthd = apply_rotary_pos_emb(q_bthd, k_bthd, cos, sin)

        if self.backend == AttnBackend.SDPA:
            # SDPA expects q,k,v as [B, H, T, d]
            q_bhtd = q_bthd.permute(0, 2, 1, 3)
            k_bhtd = k_bthd.permute(0, 2, 1, 3)
            v_bhtd = v_bthd.permute(0, 2, 1, 3)

            out_bhtd = F.scaled_dot_product_attention(
                q_bhtd, k_bhtd, v_bhtd, is_causal=is_causal, dropout_p=p
            )  # [B, H, T, d]

            # Unflatten heads back to [B, T, K, N, d] without extra transposes/copies.
            out_bkntd = out_bhtd.reshape(B, K, N, T, d)          # [B, K, N, T, d]
            out_btknd = out_bkntd.permute(0, 3, 1, 2, 4)         # [B, T, K, N, d]
            return out_btknd

        if self.backend == AttnBackend.FLASH3:
            # FlashAttn expects q,k,v as [B, T, H, d]
            out_bthd = self._flash_attn_func(
                q_bthd.contiguous(), k_bthd.contiguous(), v_bthd.contiguous(),
                dropout_p=p, causal=is_causal,
            )
            return out_bthd.reshape(B, T, K, N, d)

        # FLASH3_QKVPACKED: qkvpacked is [B, T, 3, H, d]
        qkvpacked_bt3hd = torch.stack((q_bthd, k_bthd, v_bthd), dim=2).contiguous()  # [B,T,3,H,d]
        out_bthd = self._flash_qkvpacked_func(qkvpacked_bt3hd, dropout_p=p, causal=is_causal)
        return out_bthd.reshape(B, T, K, N, d)


# =============================================================================
# Morpher
# =============================================================================
class Morpher(nn.Module):
    """
    The full model.

    Parameters:
      io_dim: input/output feature dim
      d:      per-slot dim
      time_scales: list of periods; K=len(time_scales), N=lcm(time_scales), S=sum(time_scales)
      enc_dec_rank: LoRA rank used in encoder/decoder bottlenecks
    """

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
        # --- RoPE config ---
        use_rope: bool = True,
        max_position_embeddings: int = 2048,
        rope_base: int = 10000,
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

        # K = number of scales
        self.K = len(self.time_scales)

        # d = slot feature width
        self.d = d
        if use_rope:
            assert self.d % 2 == 0, "RoPE requires even head_dim (d)"

        # N = number of streams (also the phase period)
        self.N = lcm_list(self.time_scales)

        # S = total slots per stream (sum of scale periods)
        self.S = sum(self.time_scales)

        # D = stream state width
        self.D = self.S * self.d

        # Precomputed phase tables
        self.tables = PhaseTables(self.time_scales, self.stream_head_assignment)

        # Encoder / Decoder
        self.encoder = StreamLoRAEncoder(io_dim, self.D, self.N, enc_dec_rank)
        self.decoder = StreamLoRADecoder(self.N, self.D, io_dim, enc_dec_rank)

        # Mixer (per-stream MLP over full state)
        self.mixer_expansion = mixer_expansion
        self.ln_mixer = NoParamRMSNorm(self.D)
        self.mixer = SwiGLU(self.D, self.mixer_expansion)

        # Attention input dim depends on scope
        if self.head_input_scope == HeadInputScope.SLOT:
            head_in_dim = self.d
        elif self.head_input_scope == HeadInputScope.SCALES:
            head_in_dim = self.K * self.d
        else:  # STREAM
            head_in_dim = self.D

        self.head_input_dim = head_in_dim
        self.ln_attn = NoParamRMSNorm(self.head_input_dim)

        # QKV projector produces [B, T, K, N, 3d]
        self.projector = QKVProjector(
            K=self.K,
            N=self.N,
            d=self.d,
            in_dim=self.head_input_dim,
            assignment=self.stream_head_assignment,
            # For SHARED + stream/scales scope, vectorizing across K means materializing
            # x_btkni=[B,T,K,N,in]. We only do that if it stays under this budget.
            max_shared_xhead_bytes=128 * 1024 * 1024,
        )

        # Attention backend adapter
        self.attn = AttentionAdapter(attn_backend)

        # --- Rotary positional embedding (RoPE) ---
        self.use_rope = bool(use_rope)
        self.rope = None
        if self.use_rope:
            # Cached cos/sin buffers sized to max_position_embeddings.
            self.rope = RotaryEmbedding(
                dim=self.d,
                max_position_embeddings=int(max_position_embeddings),
                base=int(rope_base),
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.mixer.reset_parameters()
        self.projector.reset_parameters()

    # -------------------------------------------------------------------------
    # One micro-step
    # -------------------------------------------------------------------------
    def step(self, z_btnd: torch.Tensor, t: int, is_causal: bool) -> torch.Tensor:
        """
        One micro-step update.

        Input:
          z_btnd : [B, T, N, D]

        Output:
          z_next_btnd : [B, T, N, D]
        """
        B, T, N, D = z_btnd.shape
        assert (N, D) == (self.N, self.D)

        phase = t % self.N
        active_slots_k: torch.Tensor = self.tables.active_slots[phase]  # [K] (slot indices)

        # ---- 1) Mixer (per-stream)
        z_mixed_btnd = z_btnd + self.dropout_mixer(self.mixer(self.ln_mixer(z_btnd)))  # [B, T, N, D]

        # ---- 2) Slot view + select active slots
        z_slots_btnsd = z_mixed_btnd.view(B, T, N, self.S, self.d)              # [B, T, N, S, d]
        z_active_btnkd = z_slots_btnsd.index_select(dim=3, index=active_slots_k)  # [B, T, N, K, d]

        # We keep attention pipeline in scale-major layout for consistency and efficiency:
        z_active_btknd = z_active_btnkd.permute(0, 1, 3, 2, 4)  # [B, T, K, N, d] (view)

        # ---- 3) Build QKV (always returns [B, T, K, N, 3d])
        if self.head_input_scope == HeadInputScope.SLOT:
            # LN over last dim (d) per active slot
            x_btnkd = self.ln_attn(z_active_btnkd)                  # [B, T, N, K, d]
            x_btknd = x_btnkd.permute(0, 1, 3, 2, 4)                # [B, T, K, N, d]

            qkv_btk_n3d = self.projector.project_slot(
                x_btknd,
                phase=phase,
                stream_to_head=self.tables.stream_to_head,
                head_to_stream=self.tables.head_to_stream,
            )

        elif self.head_input_scope == HeadInputScope.SCALES:
            # Concat active slots per stream: [B, T, N, K*d]
            x_btni = z_active_btnkd.reshape(B, T, N, self.K * self.d)  # [B, T, N, K*d]
            x_btni = self.ln_attn(x_btni)

            qkv_btk_n3d = self.projector.project_stream(
                x_btni,
                phase=phase,
                stream_to_head=self.tables.stream_to_head,
                head_to_stream=self.tables.head_to_stream,
            )

        else:  # STREAM
            # Full stream state per head input
            x_btni = self.ln_attn(z_mixed_btnd)  # [B, T, N, D]

            qkv_btk_n3d = self.projector.project_stream(
                x_btni,
                phase=phase,
                stream_to_head=self.tables.stream_to_head,
                head_to_stream=self.tables.head_to_stream,
            )

        # ---- 4) Attention over streams/scales
        # RoPE cos/sin: [T, d] (slice from cached buffers)
        cos = sin = None
        if self.use_rope:
            assert self.rope is not None
            cos_full, sin_full = self.rope()          # [max_pos, d]
            # NOTE: this applies RoPE based on absolute token index 0..T-1.
            # If you want micro-step-dependent offset, slice with (t_offset:t_offset+T) instead.
            cos = cos_full[:T]
            sin = sin_full[:T]

        out_btknd = self.attn(
            qkv_btk_n3d,
            is_causal=is_causal,
            dropout_p=self.dropout_p,
            cos=cos,
            sin=sin,
        )  # [B, T, K, N, d]
        out_btknd = self.dropout_attn(out_btknd)

        # Residual on active slots
        out_btknd = out_btknd + z_active_btknd  # [B, T, K, N, d]

        # ---- 5) Write back only active slots into the full state
        # Convert back to stream-major active-slot layout for index_copy_:
        out_btnkd = out_btknd.permute(0, 1, 3, 2, 4)  # [B, T, N, K, d]

        # Clone full state once, then update only the active slots.
        z_next_btnd = z_mixed_btnd.clone()
        z_next_slots_btnsd = z_next_btnd.view(B, T, N, self.S, self.d)
        z_next_slots_btnsd.index_copy_(dim=3, index=active_slots_k, source=out_btnkd)

        return z_next_btnd

    # -------------------------------------------------------------------------
    # Cycle update
    # -------------------------------------------------------------------------
    def forward_cycle(self, z_btnd: torch.Tensor, t0: int, is_causal: bool) -> torch.Tensor:
        """
        Run one full phase cycle of length N micro-steps.
        """
        for dt in range(self.N):
            z_btnd = self.step(z_btnd, t=t0 + dt, is_causal=is_causal)
        return z_btnd

    # -------------------------------------------------------------------------
    # Forward (burn-in + graft + short unroll)
    # -------------------------------------------------------------------------
    def forward(self, x_bti: torch.Tensor, R: int, is_causal: bool, grad_cycles: int = 1) -> torch.Tensor:
        """
        Forward pass with optional burn-in.

        Args:
          x_bti: [B, T, io_dim]
          R: number of cycles total
          grad_cycles: how many final cycles to backprop through (>=1, <=R)

        Strategy:
          - Encode to z0 (with grad)
          - Burn-in for (R - grad_cycles) cycles without autograd graph
          - "Gradient graft": keep burned-in value, but attach gradients as if starting from z0
          - Run grad_cycles cycles with autograd
          - Decode to output
        """
        assert 1 <= grad_cycles <= R
        burn_cycles = R - grad_cycles

        # Encoder must have grad so stacking works
        z0_btnd = self.encoder(x_bti)  # [B, T, N, D] with grad
        z_btnd = z0_btnd.detach()

        t = 0
        if burn_cycles > 0:
            # Burn-in without graph (fast + low memory)
            with torch.inference_mode():
                for _ in range(burn_cycles):
                    z_btnd = self.forward_cycle(z_btnd, t0=t, is_causal=is_causal)
                    t += self.N

        # Gradient graft:
        #   - value comes from burned-in z_btnd (no-grad)
        #   - gradient flows from z0_btnd
        z_btnd = z_btnd.detach() + (z0_btnd - z0_btnd.detach())

        # Final cycles with grad
        for _ in range(grad_cycles):
            z_btnd = self.forward_cycle(z_btnd, t0=t, is_causal=is_causal)
            t += self.N

        return self.decoder(z_btnd)


# =============================================================================
# Example / sanity test
# =============================================================================
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    B, T, io_dim = 4, 10, 16
    time_scales = [1, 2, 4]
    d = 12
    mixer_expansion = 4.0
    enc_dec_rank = 8

    model = Morpher(
        io_dim=io_dim,
        d=d,
        time_scales=time_scales,
        enc_dec_rank=enc_dec_rank,
        mixer_expansion=mixer_expansion,
        stream_head_assignment=StreamHeadAssignment.PRIVATE,
        head_input_scope=HeadInputScope.SLOT,  # try STREAM / SCALES too
        attn_backend=AttnBackend.SDPA,         # try FLASH3 / FLASH3_QKVPACKED if available
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
