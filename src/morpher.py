from typing import List
import torch
import torch.nn as nn
from enum import StrEnum
from math import gcd
from functools import reduce
import torch.nn.functional as F


def lcm(a, b): return a * b // gcd(a, b)
def lcm_list(xs): return reduce(lcm, xs, 1)


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
            mixer_hidden_dim: int,
            stream_head_assignment: StreamHeadAssignment = StreamHeadAssignment.SHARED,
            head_input_scope: HeadInputScope = HeadInputScope.STREAM,
            dropout: float = 0.0
        ):

        super().__init__()
        time_scales = sorted(time_scales)
        assert stream_head_assignment in StreamHeadAssignment, f"Invalid stream_head_assignment: {stream_head_assignment}"
        assert time_scales[0] == 1, "include scale=1 as fastest"
        self.time_scales = time_scales
        self.stream_head_assignment = stream_head_assignment
        self.head_input_scope = head_input_scope

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


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

        # Encodes input to initial state of each stream
        self.enc_base = nn.Linear(io_dim, self.D, bias=False)

        # Embedding for each stream to break the symmetry so each
        # stream has a different initial state
        self.stream_bias = nn.Embedding(self.N, self.D)

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
        self.Wq = nn.Parameter(torch.randn(self.H, self.head_input_dim, self.d) * (self.d ** -0.5))
        self.Wk = nn.Parameter(torch.randn(self.H, self.head_input_dim, self.d) * (self.d ** -0.5))
        self.Wv = nn.Parameter(torch.randn(self.H, self.head_input_dim, self.d) * (self.d ** -0.5))

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
        slot_offset = 0
        for phase in range(self.N):
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

    def encode(self, x):
        """
        Encoder input tokens into initial multi-stream multi-slot state
        x: [B, T, io_dim]
        """

        B, T, _ = x.shape

        # [B, T, D]
        base = self.enc_base(x)

        # repeat across streams: [N, B, T, D]
        z0 = base.unsqueeze(0).repeat(self.N, 1, 1, 1)

        # [N, D]
        stream_bias = self.stream_bias(torch.arange(self.N, device=x.device))

        # Add the embeddings to the input
        z0 = z0 + stream_bias[:, None, None, :]
        
        return z0

    def step(self, z: torch.Tensor, t: int, is_causal: bool):
        """
        z: [N, B, T, D]
        """

        N, B, T, _ = z.shape
        phase = t % self.N

        # Apply mixer and residual connection -> [N, B, T, D]
        z_head = z + self.dropout(self.mixer(self.ln_mixer(z)))

        # Choose active slots for this phase -> [K] 
        active_slots = self.active_slots[phase]

        # split into slots -> [N, B, T, S, d]
        z_slots = z_head.view(N, B, T, self.S, self.d)

        # Select active slots -> [N, B, T, K, d]
        z_active = z_slots[:, :, :, active_slots]

        # Choose active heads for this phase -> [N, K]
        active_heads = self.head_assignments[:, active_slots]
        
        Wk = self.Wk[active_heads] # [N, K, head_input_dim, d]
        Wq = self.Wq[active_heads] # [N, K, head_input_dim, d]
        Wv = self.Wv[active_heads] # [N, K, head_input_dim, d]

        if self.head_input_scope == HeadInputScope.STREAM:
            # head_input_dim = D
            z_attn = self.ln_attn(z_head)
            keys = torch.einsum('nbth,nkhd->nbtkd', z_attn, Wk)
            queries = torch.einsum('nbth,nkhd->nbtkd', z_attn, Wq)
            values = torch.einsum('nbth,nkhd->nbtkd', z_attn, Wv)

        else:

            if self.head_input_scope == HeadInputScope.SCALES:
                # head_input_dim = d * K
                z_attn = self.ln_attn(z_active.view(N, B, T, -1))
                keys = torch.einsum('nbth,nkhd->nbtkd', z_attn, Wk)
                queries = torch.einsum('nbth,nkhd->nbtkd', z_attn, Wq)
                values = torch.einsum('nbth,nkhd->nbtkd', z_attn, Wv)

            elif self.head_input_scope == HeadInputScope.SLOT:
                # head_input_dim = d
                z_attn = self.ln_attn(z_active) # [N, B, T, K, d]
                keys = torch.einsum('nbtkh,nkhd->nbtkd', z_attn, Wk)
                queries = torch.einsum('nbtkh,nkhd->nbtkd', z_attn, Wq)
                values = torch.einsum('nbtkh,nkhd->nbtkd', z_attn, Wv)

        # Reshape to SDPA format: [N, B, T, K, d] -> [B, N*K, T, d]
        # Permute to [B, N, T, K, d] then flatten N*K -> [B, P, T, d]
        keys = keys.permute(1, 0, 2, 3, 4).contiguous().view(B, self.N * self.K, T, self.d)
        queries = queries.permute(1, 0, 2, 3, 4).contiguous().view(B, self.N * self.K, T, self.d)
        values = values.permute(1, 0, 2, 3, 4).contiguous().view(B, self.N * self.K, T, self.d)
        attn_out = F.scaled_dot_product_attention(queries, keys, values, is_causal=is_causal)  # [B,P,T,d]
        attn_out = attn_out.view(B, self.N, self.K, T, self.d).permute(1, 0, 3, 2, 4).contiguous() # [N, B, T, K, d]
        attn_out = attn_out + z_active

        z_next_slots = z_slots.clone()  # [N, B, T, S, d]
        z_next_slots.index_copy_(dim=3, index=active_slots, source=attn_out)

        z_next = z_next_slots.view(N, B, T, self.D)

        return z_next


    def forward(self, x, R: int, is_causal: bool):
        z = self.encode(x)

        T = self.N

        z = self.step(z=z, t=0, is_causal=is_causal)

        return z



if __name__ == "__main__":

    # create a random input for the transformer module
    T = 10
    B = 4
    head_dim = 8
    dim_io = 16
    time_scales = [1, 2, 4]
    mixer_dim = 72



    d = 12
    model = Morpher(dim_io, d, time_scales, mixer_dim, stream_head_assignment=StreamHeadAssignment.PRIVATE)
    print(model)

    x = torch.randn(B, T, dim_io)
    
    print(model(x, R=1, is_causal=True))

    
