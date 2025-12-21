from typing import List
import torch
import torch.nn as nn
from enum import StrEnum
from math import gcd
from functools import reduce
import torch.nn.functional as F


def lcm(a, b): return a * b // gcd(a, b)
def lcm_list(xs): return reduce(lcm, xs, 1)


class StreamLoRAEncoder(nn.Module):
    """
    Projects input features to multiple independent streams using LoRA adapters.

    Architecture:
    - Shared base projection: input_dim -> output_dim
    - Per-stream LoRA adapters: each stream has its own low-rank B matrix
    - Zero-initialized B matrices ensure stable training

    This enables each stream to learn specialized adaptations while sharing
    a common base representation.

    Interface:
      x:  [B, T, input_dim]  # Input features
      z0: [B, T, N, output_dim]  # Batch-first with N stream representations
    """

    def __init__(self, input_dim: int, output_dim: int, num_splits: int, rank: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = num_splits
        self.rank = rank

        # Shared base projection
        self.enc_base = nn.Linear(input_dim, output_dim, bias=False)

        # Shared low-rank "A" (input_dim -> rank)
        self.enc_A = nn.Linear(input_dim, rank, bias=False)

        # Per-stream low-rank "B_n" (rank -> output_dim), zero-initialized for stable training
        self.enc_B = nn.Parameter(torch.zeros(rank, num_splits, output_dim))

        # Learned scalar gate for the adapter branch (small init to enable gradient flow)
        self.enc_beta = nn.Parameter(torch.tensor(0.01))

        self.reset_parameters()

    def reset_parameters(self):
        # Default init for Linear layers is typically fine; explicitly reset anyway.
        nn.init.xavier_uniform_(self.enc_base.weight)
        nn.init.xavier_uniform_(self.enc_A.weight)

        # B matrices: zero-init so adapters start disabled (standard LoRA)
        nn.init.zeros_(self.enc_B)

        # Beta gate: small init (0.01) to enable gradient flow while keeping initial effect minimal
        with torch.no_grad():
            self.enc_beta.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, T, input_dim]
        z0: [B, T, N, output_dim]
        """
        # Shared base: [B,T,output_dim]
        base = self.enc_base(x)

        # Shared low-rank features: [B,T,rank]
        h = self.enc_A(x)

        # Per-stream B matrices: [N, rank, output_dim]
        Bmat = self.enc_B

        # Low-rank delta per stream: einsum over rank -> [B,T,N,output_dim]
        # delta[b,t,n,d] = sum_r h[b,t,r] * Bmat[r,n,d]
        delta = torch.einsum("b t r, r n d -> b t n d", h, Bmat)

        # Apply learned scalar gate
        z0 = base.unsqueeze(2) + self.enc_beta * delta  # [B,T,N,output_dim]

        return z0



class StreamLoRADecoder(nn.Module):
    """
    Stream-conditioned LoRA decoder that combines multi-stream representations.

    Interface:
      z: [B, T, N, input_dim]   # Multi-stream representations
      y: [B, T, output_dim]     # Decoded output

    Method:
      1. Concatenate all streams: [B,T,N,input_dim] -> [B,T,N*input_dim]
      2. Low-rank projection: (N*input_dim) -> rank -> output_dim

    This factorizes a large (N*input_dim)->output_dim matrix into
    (N*input_dim)->rank followed by rank->output_dim for parameter efficiency.
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
        self.proj_down = nn.Linear(in_dim, rank, bias=bias)   # U
        self.proj_up = nn.Linear(rank, output_dim, bias=bias) # V

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier is a good default for these projections
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, T, N, input_dim]
        y: [B, T, output_dim]
        """
        assert z.dim() == 4, f"Expected z with shape [B,T,N,input_dim], got {z.shape}"
        B, T, N, D = z.shape
        assert N == self.N, f"Expected N={self.N}, got N={N}"
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got D={D}"

        # [B,T,N,input_dim] -> [B,T,N*input_dim]
        cat = z.contiguous().view(B, T, N * D)

        # Low-rank projection: (N*input_dim)->r->output_dim
        h = self.proj_down(self.ln(cat))   # [B,T,r]
        y = self.proj_up(h)                # [B,T,output_dim]
        return y


class StreamHeadAssignment(StrEnum):
    SHARED = "shared"       # A head is shared across streams 
    PRIVATE = "private"     # Each stream has it's own set of heads

class HeadInputScope(StrEnum):
    SLOT = "slot"           # A head is a function of it's slot in that stream per step                  
    STREAM = "stream"       # A head is a function of full stream state per step
    SCALES = "scales"       # A head is a function of active scales per step



class Morpher(nn.Module):
    def __init__(self,
            io_dim: int,
            d: int,
            time_scales: List[int],
            enc_dec_rank: int,
            mixer_hidden_dim: int,
            stream_head_assignment: StreamHeadAssignment = StreamHeadAssignment.SHARED,
            head_input_scope: HeadInputScope = HeadInputScope.STREAM,
            dropout: float = 0.0,
        ):

        super().__init__()
        time_scales = sorted(time_scales)
        assert stream_head_assignment in StreamHeadAssignment, f"Invalid stream_head_assignment: {stream_head_assignment}"
        assert time_scales[0] == 1, "include scale=1 as fastest"
        self.time_scales = time_scales
        self.stream_head_assignment = stream_head_assignment
        self.head_input_scope = head_input_scope
        self.dropout_p = dropout

        self.dropout_mixer = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.dropout_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


        # K: Number of active slots/heads per stream per step
        self.K = len(time_scales) 

        # d: Dimension of each slot
        self.d = d

        # N: Period or number of streams
        self.N = lcm_list(time_scales)

        # S: Number of slots in each stream
        self.S = sum(self.time_scales)

        # D: Dimension of state of each stream
        self.D = self.d * self.S

        # Stream-conditioned LoRA encoder for input to multi-stream representations
        self.encoder = StreamLoRAEncoder(
            input_dim=io_dim,
            output_dim=self.D,
            num_splits=self.N,
            rank=enc_dec_rank
        )

        # Stream State Mixer
        self.ln_mixer = nn.LayerNorm(self.D)
        self.mixer = nn.Sequential(
            nn.Linear(self.D, mixer_hidden_dim),
            nn.GELU(),
            nn.Linear(mixer_hidden_dim, self.D),
        )

        # Attention Heads
        self.H = self.K * self.N
        if self.head_input_scope == HeadInputScope.SLOT:
            self.head_input_dim = self.d
        elif self.head_input_scope == HeadInputScope.STREAM:
            self.head_input_dim = self.D
        elif self.head_input_scope == HeadInputScope.SCALES:
            self.head_input_dim = self.d * self.K
        self.ln_attn = nn.LayerNorm(self.head_input_dim)
        self.Wq = nn.Parameter(torch.randn(self.H, self.head_input_dim, self.d))
        self.Wk = nn.Parameter(torch.randn(self.H, self.head_input_dim, self.d))
        self.Wv = nn.Parameter(torch.randn(self.H, self.head_input_dim, self.d))
        
        # Stream-conditioned LoRA decoder for multi-stream representations to output
        self.decoder = StreamLoRADecoder(
            num_splits=self.N,
            input_dim=self.D,
            output_dim=io_dim,
            rank=enc_dec_rank
        )

        self._build_active_slots_table()
        self._build_head_assignments_table()

    def _build_active_slots_table(self):
        """
        S: sum(scales) = total number of slots in a state 
        K: len(scales) = number of active slots per step 

        There is one active slot per scale and which slots are active at each step is only a function of the step phase. 

        scales=[1, 2, 4]
        N (streams/period) = 4 
        K (scales) = 3
        total_slots=sum(scales) = 7

        === active_slots[phase, scale] table === 
        phase | s=1 s=2 s=4
          0   | 0   1   3 
          1   | 0   2   4 
          2   | 0   1   5 
          3   | 0   2   6
        """
        active_slots = torch.empty(self.N, self.K, dtype=torch.long)
        for phase in range(self.N):
            slot_offset = 0
            for scale_idx, scale in enumerate(self.time_scales):
                active_slots[phase, scale_idx] = slot_offset + (phase % scale)
                slot_offset += scale

        # [N, K] \in [0, S)
        self.register_buffer("active_slots", active_slots, persistent=False)

    def _build_head_assignments_table(self):
        """
        Each slot in a stream is mapped to one of the N*K heads. 

        head_assignments[stream_id, slot_id] -> head_id

        S: sum(scales) = total number of slots in a state 
        K: len(scales) = number of active slots per step
        N: number of streams
        H: number of heads = K * N
        

        stream_head_assignment modes:
            "shared": slot-wise crossing (phase-slot j chooses a different head permutation)
            "private": stream n always uses head n within each scale for all slots. Each stream has it's own set of heads


        Scales [1, 2, 4] (N=4, K=3, H=12) 
        E.g Shared
        === head_assignments[stream_id, slot_id] table ===
        stream_id | 0   1   2   3   4   5   6
        ----------|-----------------------------
          0       | 0   4   6   8   9   10  11
          1       | 1   5   7   9   10  11  8
          2       | 2   6   4   10  11  8   9
          3       | 3   7   5   11  8   9   10

        E.g Private
        === head_assignments[stream_id, slot_id] table ===
        stream_id | 0   1   2   3   4   5   6
        ----------|-----------------------------
          0       | 0   4   4   8   8   8   8
          1       | 1   5   5   9   9   9   9
          2       | 2   6   6  10  10  10  10
          3       | 3   7   7  11  11  11  11
        """

        head_assignments = torch.empty(self.N, self.S, dtype=torch.long) # [N, S]
        stream_ids = torch.arange(self.N, dtype=torch.long)                         # [N]

        scale_offset = 0
        for scale_idx, s in enumerate(self.time_scales):
            alpha = self.N // s 
            # slots for this scale correspond to phase-slot index j = 0..s-1
            for j in range(s):  # scale 1 has 1 slot, scale 2 has 2 slots, scale 4 has 4 slots, etc.
                if self.stream_head_assignment == StreamHeadAssignment.PRIVATE:
                    head_ids = stream_ids  # [N]
                elif self.stream_head_assignment == StreamHeadAssignment.SHARED:
                    head_ids = (stream_ids + alpha * j) % self.N  # [N]
                else:
                    raise ValueError("stream_head_assignment must be 'private' or 'shared'")

                global_ids = scale_idx * self.N + head_ids  # [N] in [0..K*N-1]
                head_assignments[:, scale_offset + j] = global_ids

            scale_offset += s
        self.register_buffer("head_assignments", head_assignments, persistent=False)


    def step(self, z: torch.Tensor, t: int, is_causal: bool):
        """
        z: [B, T, N, D]
        """

        B, T, N, _ = z.shape

        phase = t % self.N

        # Apply mixer and residual connection -> [B, T, N, D]
        z_head = z + self.dropout_mixer(self.mixer(self.ln_mixer(z)))

        # Choose active slots for this phase -> [K] 
        active_slots = self.active_slots[phase]

        # split into slots -> [B, T, N, S, d]
        z_slots = z_head.view(B, T, N, self.S, self.d)

        # Select active slots -> [B, T, N, K, d]
        z_active = z_slots[:, :, :, active_slots, :]  # [B, T, N, K, d]

        # Choose active heads for this phase -> [N, K]
        active_heads = self.head_assignments[:, active_slots]
        
        Wk = self.Wk[active_heads] # [N, K, head_input_dim, d]
        Wq = self.Wq[active_heads] # [N, K, head_input_dim, d]
        Wv = self.Wv[active_heads] # [N, K, head_input_dim, d]

        if self.head_input_scope == HeadInputScope.STREAM:
            # head_input_dim = D
            z_attn = self.ln_attn(z_head)  # [B, T, N, D]
            keys = torch.einsum('btnh,nkhd->bnktd', z_attn, Wk)  # [B, N, K, T, d]
            queries = torch.einsum('btnh,nkhd->bnktd', z_attn, Wq)  # [B, N, K, T, d]
            values = torch.einsum('btnh,nkhd->bnktd', z_attn, Wv)  # [B, N, K, T, d]

        else:

            if self.head_input_scope == HeadInputScope.SCALES:
                # head_input_dim = d * K
                z_attn = self.ln_attn(z_active.view(B, T, N, -1))  # [B, T, N, d*K]
                keys = torch.einsum('btnh,nkhd->bnktd', z_attn, Wk)  # [B, N, K, T, d]
                queries = torch.einsum('btnh,nkhd->bnktd', z_attn, Wq)  # [B, N, K, T, d]
                values = torch.einsum('btnh,nkhd->bnktd', z_attn, Wv)  # [B, N, K, T, d]

            elif self.head_input_scope == HeadInputScope.SLOT:
                # head_input_dim = d
                z_attn = self.ln_attn(z_active)  # [B, T, N, K, d]
                keys = torch.einsum('btnkh,nkhd->bnktd', z_attn, Wk)  # [B, N, K, T, d]
                queries = torch.einsum('btnkh,nkhd->bnktd', z_attn, Wq)  # [B, N, K, T, d]
                values = torch.einsum('btnkh,nkhd->bnktd', z_attn, Wv)  # [B, N, K, T, d]

        # Reshape to SDPA format: [B, N, K, T, d] -> [B, N*K, T, d]
        keys = keys.reshape(B, N * self.K, T, self.d)  # [B, N*K, T, d]
        queries = queries.reshape(B, N * self.K, T, self.d)  # [B, N*K, T, d]
        values = values.reshape(B, N * self.K, T, self.d)  # [B, N*K, T, d]
        attn_out = F.scaled_dot_product_attention(  # [B, N*K, T, d]
            queries, keys, values, 
            is_causal=is_causal, 
            dropout_p=self.dropout_p if self.training else 0.0
        )  
        attn_out = self.dropout_attn(attn_out)
        attn_out = attn_out.permute(0, 2, 1, 3).view(B, T, N, self.K, self.d)  # [B, T, N, K, d]
        attn_out = attn_out + z_active

        # Update active slots with attention output (in-place)
        z_slots[:, :, :, active_slots, :] = attn_out  # [B, T, N, K, d]

        z_next = z_slots.view(B, T, N, self.D)

        return z_next


    def forward(self, x, R: int, is_causal: bool):
        # Encoder outputs [B, T, N, D] - step function now handles this format
        z = self.encoder(x)

        z = self.step(z=z, t=0, is_causal=is_causal)

        # Decoder converts [B, T, N, D] back to [B, T, io_dim]
        y = self.decoder(z)

        return y



if __name__ == "__main__":

    # create a random input for the transformer module
    T = 10
    B = 4
    head_dim = 8
    dim_io = 16
    time_scales = [1, 2, 4]
    mixer_dim = 72

    d = 12
    enc_dec_rank = 8
    model = Morpher(dim_io, d, time_scales, enc_dec_rank, mixer_dim, stream_head_assignment=StreamHeadAssignment.PRIVATE)
    print(model)

    x = torch.randn(B, T, dim_io)
    
    y = model(x, R=1, is_causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    
