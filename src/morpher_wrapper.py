"""
Morpher wrapper for HRM training pipeline integration.

This module provides wrapper classes that adapt the Morpher model to be compatible
with HRM's training infrastructure, including ACT (Adaptive Computation Time) support.

Classes:
    MorpherInnerCarry: Dataclass for internal state management
    MorpherACTCarry: Full ACT state including halting info
    MorpherInner: Core wrapper around Morpher model
    MorpherACT: ACT wrapper with adaptive halting
    MorpherFixedSteps: Fixed-step wrapper (no ACT)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

# Import from local morpher
from morpher import Morpher, StreamHeadAssignment, HeadInputScope, AttnBackend
from utils import trunc_normal_init_, CastedLinear, lcm_list

# Import HRM components (these will be added to path at runtime)
# from external HRM: models.layers, models.sparse_embedding, etc.


# =============================================================================
# Configuration
# =============================================================================
class MorpherWrapperConfig(BaseModel):
    """Configuration for Morpher wrapper, compatible with HRM training."""
    
    # Required by HRM training infrastructure
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    
    # Morpher architecture
    d: int = 64                          # per-slot dimension
    time_scales: List[int] = [1, 2, 4]   # multi-scale slots
    enc_dec_rank: int = 32               # LoRA rank for encoder/decoder
    mixer_expansion: float = 4.0
    dropout: float = 0.0
    
    # I/O dimension (for embeddings, decoder output)
    io_dim: int = 512
    
    # Stacking (more depth, not hierarchy)
    num_levels: int = 1
    cycles_per_level: List[int] = [2]    # cycles per ACT step, per level
    
    # ACT settings
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    
    # Attention and position encoding
    stream_head_assignment: str = "shared"
    head_input_scope: str = "slot"
    attn_backend: str = "sdpa"
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    
    # Precision
    forward_dtype: str = "bfloat16"
    
    # Grad settings
    grad_cycles: int = 1  # How many final cycles to backprop through


# =============================================================================
# Carry dataclasses
# =============================================================================
@dataclass
class MorpherInnerCarry:
    """Internal state for a single Morpher level."""
    z_btnd: torch.Tensor  # [B, T, N, D] - multi-stream state


@dataclass
class MorpherStackedCarry:
    """State for multiple stacked Morpher levels."""
    level_states: List[MorpherInnerCarry]  # One per level


@dataclass
class MorpherACTCarry:
    """Full ACT state including halting information."""
    inner_carry: MorpherStackedCarry
    
    steps: torch.Tensor       # [B] - number of ACT steps taken
    halted: torch.Tensor      # [B] - bool, whether each sample has halted
    
    current_data: Dict[str, torch.Tensor]  # Current batch data


# =============================================================================
# Embedding layers (matching HRM's approach)
# =============================================================================
class CastedEmbedding(nn.Module):
    """Embedding with truncated normal init and dtype casting."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


# =============================================================================
# MorpherInner - Core wrapper
# =============================================================================
class MorpherInner(nn.Module):
    """
    Core wrapper around Morpher model for HRM compatibility.
    
    Handles:
    - Token and puzzle embeddings
    - Running cycles and producing outputs
    - Q-head for ACT halting decisions
    """
    
    def __init__(self, config: MorpherWrapperConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Calculate derived dimensions
        self.N = lcm_list(config.time_scales)
        self.S = sum(config.time_scales)
        self.D = self.S * config.d
        
        # Embedding scale (like HRM)
        self.embed_scale = math.sqrt(config.io_dim)
        embed_init_std = 1.0 / self.embed_scale
        
        # Token embedding
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.io_dim, 
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        
        # Puzzle embedding (sparse, optional)
        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.io_dim)  # ceil div
        self.puzzle_emb = None
        if config.puzzle_emb_ndim > 0:
            # We'll use a simple embedding for now (HRM uses sparse)
            self.puzzle_emb = CastedEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                init_std=0, cast_to=self.forward_dtype  # zero init
            )
        
        # Position embeddings (learned, if not using RoPE in Morpher)
        if config.pos_encodings == "learned":
            total_seq_len = config.seq_len + self.puzzle_emb_len
            self.embed_pos = CastedEmbedding(
                total_seq_len, config.io_dim,
                init_std=embed_init_std, cast_to=self.forward_dtype
            )
        
        # Create Morpher level(s)
        self.morphers = nn.ModuleList()
        for level_idx in range(config.num_levels):
            morpher = Morpher(
                io_dim=config.io_dim,
                d=config.d,
                time_scales=config.time_scales,
                enc_dec_rank=config.enc_dec_rank,
                mixer_expansion=config.mixer_expansion,
                stream_head_assignment=StreamHeadAssignment(config.stream_head_assignment),
                head_input_scope=HeadInputScope(config.head_input_scope),
                attn_backend=AttnBackend(config.attn_backend),
                dropout=config.dropout,
                use_rope=(config.pos_encodings == "rope"),
                max_position_embeddings=config.seq_len + self.puzzle_emb_len + 128,
                rope_base=int(config.rope_theta),
            )
            self.morphers.append(morpher)
        
        # LM head: decoder output -> vocab logits
        self.lm_head = CastedLinear(config.io_dim, config.vocab_size, bias=False)
        
        # Q head for ACT: decoder output -> (q_halt, q_continue)
        self.q_head = CastedLinear(config.io_dim, 2, bias=True)
        
        # Initialize Q head to near-zero for faster learning
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
    
    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        """Create input embeddings from tokens and puzzle identifiers."""
        # Token embedding
        embedding = self.embed_tokens(inputs.to(torch.int32))
        
        # Puzzle embeddings (prepended)
        if self.puzzle_emb is not None and self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            # Pad to multiple of io_dim
            pad_count = self.puzzle_emb_len * self.config.io_dim - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.io_dim)
            embedding = torch.cat((puzzle_embedding, embedding), dim=1)
        
        # Position embeddings (for learned mode)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        
        return self.embed_scale * embedding
    
    def empty_carry(self, batch_size: int) -> MorpherStackedCarry:
        """Create empty carry state (to be filled on first forward)."""
        total_seq_len = self.config.seq_len + self.puzzle_emb_len
        level_states = []
        for morpher in self.morphers:
            z = torch.empty(
                batch_size, total_seq_len, morpher.N, morpher.D,
                dtype=self.forward_dtype
            )
            level_states.append(MorpherInnerCarry(z_btnd=z))
        return MorpherStackedCarry(level_states=level_states)
    
    def reset_carry(
        self, 
        reset_flag: torch.Tensor, 
        carry: MorpherStackedCarry,
        input_embeddings: torch.Tensor
    ) -> MorpherStackedCarry:
        """
        Reset carry for samples that have halted.
        
        For Morpher, we re-encode from input embeddings (no learned init buffers).
        """
        new_level_states = []
        
        # Get fresh encoded state from first morpher's encoder
        # (all morphers share same io_dim)
        fresh_z = self.morphers[0].encoder(input_embeddings)  # [B, T, N, D]
        
        for level_idx, (morpher, level_state) in enumerate(zip(self.morphers, carry.level_states)):
            if level_idx == 0:
                # First level uses input
                new_z = torch.where(
                    reset_flag.view(-1, 1, 1, 1),
                    fresh_z,
                    level_state.z_btnd
                )
            else:
                # Subsequent levels: also reset to encoder output (or could be zeros)
                fresh_z_level = morpher.encoder(input_embeddings)
                new_z = torch.where(
                    reset_flag.view(-1, 1, 1, 1),
                    fresh_z_level,
                    level_state.z_btnd
                )
            new_level_states.append(MorpherInnerCarry(z_btnd=new_z))
        
        return MorpherStackedCarry(level_states=new_level_states)
    
    def forward(
        self,
        carry: MorpherStackedCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[MorpherStackedCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run one ACT step (multiple complete cycles per level).
        
        Returns:
            new_carry: Updated state
            logits: [B, seq_len, vocab_size] - LM predictions
            (q_halt, q_continue): Q-values for ACT halting
        """
        # Get input embeddings
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )
        
        # Run each level
        new_level_states = []
        z_out = None  # Will hold the output of the last level
        
        for level_idx, (morpher, level_state) in enumerate(zip(self.morphers, carry.level_states)):
            z = level_state.z_btnd
            cycles = self.config.cycles_per_level[level_idx] if level_idx < len(self.config.cycles_per_level) else 1
            
            # Burn-in cycles (no grad) + final cycle(s) (with grad)
            burn_cycles = max(0, cycles - self.config.grad_cycles)
            grad_cycles = min(cycles, self.config.grad_cycles)
            
            t = 0
            
            # Re-encode on each step (input injection like HRM)
            if level_idx == 0:
                # First level: add input embeddings to state
                z_input = morpher.encoder(input_embeddings)
                z = z + z_input - z.detach() + z.detach()  # gradient graft
            else:
                # Subsequent levels: could receive info from previous level
                # For now, just use previous level's decoded output as injection
                if z_out is not None:
                    z_input = morpher.encoder(z_out)
                    z = z + z_input - z.detach() + z.detach()
            
            # Burn-in (no grad)
            if burn_cycles > 0:
                with torch.no_grad():
                    for _ in range(burn_cycles):
                        z = morpher.forward_cycle(z, t0=t, is_causal=False)
                        t += morpher.N
            
            # Cycles with grad
            for _ in range(grad_cycles):
                z = morpher.forward_cycle(z, t0=t, is_causal=False)
                t += morpher.N
            
            new_level_states.append(MorpherInnerCarry(z_btnd=z.detach()))
            
            # Decode this level's output
            z_out = morpher.decoder(z)  # [B, T, io_dim]
        
        # Remove puzzle embedding positions from output
        output = z_out[:, self.puzzle_emb_len:]  # [B, seq_len, io_dim]
        
        # LM logits
        logits = self.lm_head(output)  # [B, seq_len, vocab_size]
        
        # Q head (use first position of decoded output)
        q_input = z_out[:, 0, :]  # [B, io_dim]
        q_logits = self.q_head(q_input).to(torch.float32)  # [B, 2]
        q_halt = q_logits[..., 0]
        q_continue = q_logits[..., 1]
        
        new_carry = MorpherStackedCarry(level_states=new_level_states)
        return new_carry, logits, (q_halt, q_continue)


# =============================================================================
# MorpherACT - ACT wrapper with adaptive halting
# =============================================================================
class MorpherACT(nn.Module):
    """
    ACT wrapper around Morpher for adaptive computation time.
    
    Compatible with HRM's ACTLossHead.
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = MorpherWrapperConfig(**config_dict)
        self.inner = MorpherInner(self.config)
    
    @property
    def puzzle_emb(self):
        """Expose puzzle embedding for optimizer (like HRM)."""
        return self.inner.puzzle_emb
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MorpherACTCarry:
        """Create initial carry state from batch."""
        batch_size = batch["inputs"].shape[0]
        
        return MorpherACTCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Start halted to trigger reset
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(
        self,
        carry: MorpherACTCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[MorpherACTCarry, Dict[str, torch.Tensor]]:
        """
        One ACT step with halting logic.
        
        Returns:
            new_carry: Updated state
            outputs: Dict with logits, q_halt_logits, q_continue_logits
        """
        # Get input embeddings for reset
        input_embeddings = self.inner._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )
        
        # Reset halted sequences with new data
        new_inner_carry = self.inner.reset_carry(
            carry.halted, carry.inner_carry, input_embeddings
        )
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }
        
        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Update step count
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step
            
            # ACT halting logic (only during training with ACT enabled)
            if self.training and (self.config.halt_max_steps > 1):
                # Halt if Q says to halt
                halted = halted | (q_halt_logits > q_continue_logits)
                
                # Exploration: sometimes force more steps
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)
                
                # Compute target Q for bootstrapping
                next_inner_carry, _, (next_q_halt, next_q_continue) = self.inner(
                    new_inner_carry, new_current_data
                )
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt,
                        torch.maximum(next_q_halt, next_q_continue)
                    )
                )
        
        new_carry = MorpherACTCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )
        
        return new_carry, outputs


# =============================================================================
# MorpherFixedSteps - Fixed-step wrapper (no ACT)
# =============================================================================
class MorpherFixedSteps(nn.Module):
    """
    Fixed-step wrapper for Morpher (no adaptive halting).
    
    Runs exactly halt_max_steps ACT steps (typically set to 1).
    Compatible with HRM's ACTLossHead.
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
        # Force halt_max_steps if not in config
        config_dict.setdefault("halt_max_steps", 1)
        self.config = MorpherWrapperConfig(**config_dict)
        self.inner = MorpherInner(self.config)
    
    @property
    def puzzle_emb(self):
        """Expose puzzle embedding for optimizer."""
        return self.inner.puzzle_emb
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MorpherACTCarry:
        """Create initial carry state."""
        batch_size = batch["inputs"].shape[0]
        
        return MorpherACTCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(
        self,
        carry: MorpherACTCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[MorpherACTCarry, Dict[str, torch.Tensor]]:
        """One step without adaptive halting."""
        # Get input embeddings for reset
        input_embeddings = self.inner._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )
        
        # Reset halted sequences
        new_inner_carry = self.inner.reset_carry(
            carry.halted, carry.inner_carry, input_embeddings
        )
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }
        
        # Forward
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Fixed halting: always halt after max_steps
        new_steps = new_steps + 1
        halted = new_steps >= self.config.halt_max_steps
        
        new_carry = MorpherACTCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )
        
        return new_carry, outputs
