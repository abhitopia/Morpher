#!/usr/bin/env python3
"""
eca.py

A clean, minimal, on-the-fly 1D Elementary Cellular Automaton (ECA) dataset for PyTorch.

Design goals:
- On-the-fly generation (no stored samples)
- Deterministic per index (reproducible; multiprocessing-safe)
- User controls splits by instantiating separate datasets (no baked-in train/val/test logic)
- Fixed input/output length (L) regardless of step gap (k)
- Metadata includes rule + k (+ optional p + index)

Each sample:
  x0: [L] binary (long 0/1 by default for embedding indexing)
  xk: [L] binary (same format)
  meta: dict(rule=int, k=int, index=int, p=float)

Minimal critical config:
- rules: list[int] or int
- L: int
- p: float or (p_min, p_max)  (Bernoulli density)
- boundary: "wrap" or "zero"
- seed: int
- num_samples: virtual length for DataLoader

k values are controlled by the sampler, not the dataset. Use create_eca_dataloader() for easy setup.

Usage sketch:
  cfg = ECADatasetConfig(rules=[30, 110], L=64, p=0.5, seed=0)
  
  # Basic usage
  loader = create_eca_dataloader(cfg, batch_size=64, k_values=range(1, 9))
  
  # With curriculum (start easy, ramp up)
  curriculum = CurriculumConfig(start_k=[1, 2], end_k=list(range(1, 9)), ramp_batches=5000)
  loader = create_eca_dataloader(cfg, batch_size=64, k_values=range(1, 9), curriculum=curriculum)

Notes:
- This implements Elementary CA (radius=1, 2-state), Wolfram rule numbers 0..255.
- Rule bit ordering: neighborhood index = 4*left + 2*center + right (000..111 -> 0..7).
  Output bit is (rule >> index) & 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import hashlib

import torch
from torch.utils.data import Dataset, Sampler, DataLoader


# ---------------------------
# Handy constants (optional)
# ---------------------------
WOLFRAM_CLASS_EXAMPLES: Dict[str, List[int]] = {
    # Representative examples (heuristic; not a definitive partition of all 256 rules)
    "class_1": [0, 32, 160, 232],
    "class_2": [4, 108, 218, 250],
    "class_3": [22, 30, 126, 150, 182],
    "class_4": [54, 110],
}


# ---------------------------
# Config
# ---------------------------
PType = Union[float, Tuple[float, float]]


@dataclass
class ECADataConfig:
    """Complete configuration for ECA data loading.
    
    Contains all parameters needed to fully reproduce a data loading setup:
    - Dataset params (rules, L, p, boundary, seed, dtype)
    - Batching params (total_batches, batch_size, k_values, shuffle, drop_last)
    - Curriculum params (k_max_start, warmup, ramp)
    
    Use `create_dataloader()` method to get a configured DataLoader.
    """
    # Dataset params
    rules: Sequence[int]                 # e.g. [30] or [30, 110]
    L: int                               # width/length of the 1D state
    p: PType = 0.5                       # Bernoulli density or (p_min, p_max) sampled per example
    boundary: str = "wrap"               # "wrap" or "zero"
    seed: int = 0
    dtype: torch.dtype = torch.long      # return x0/xk as long for embedding indexing
    
    # Batching params
    total_batches: int = 10_000          # total number of batches (replaces num_samples)
    batch_size: int = 64
    k_values: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8)
    shuffle: bool = True
    drop_last: bool = False
    
    # Curriculum params: gradually increase max allowed k
    # warmup/ramp can be int (absolute batches) or float in (0,1) (fraction of total_batches)
    k_max_start: Optional[int] = None    # starting max k; if None, defaults to min(k_values)
    warmup: Union[int, float] = 0        # batches (or fraction) before curriculum starts ramping
    ramp: Union[int, float] = 0          # 0 = no curriculum; else batches (or fraction) to ramp
    
    def _resolve_batches(self, value: Union[int, float]) -> int:
        """Convert fraction to absolute batch count."""
        if isinstance(value, float) and 0 < value < 1:
            return int(value * self.total_batches)
        return int(value)
    
    def create_dataloader(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Create a fully configured DataLoader from this config.
        
        Args:
            num_workers: number of DataLoader workers (runtime setting)
            pin_memory: pin memory for faster GPU transfer
            collate_fn: optional custom collate function
        
        Returns:
            DataLoader with batch_sampler accessible via loader.batch_sampler
        
        Example:
            >>> config = ECADataConfig(rules=[30, 110], L=64, total_batches=1000)
            >>> loader = config.create_dataloader()
            >>> x0, xk, meta = next(iter(loader))
        """
        # Compute num_samples from total_batches
        num_samples = self.total_batches * self.batch_size
        
        dataset = ECADataset1D(
            rules=list(self.rules),
            L=self.L,
            p=self.p,
            boundary=self.boundary,
            seed=self.seed,
            num_samples=num_samples,
            dtype=self.dtype,
        )
        
        # Derive sampler seed from dataset seed
        sampler_seed = self.seed + 0x12345678
        
        # Resolve curriculum params
        warmup_batches = self._resolve_batches(self.warmup)
        ramp_batches = self._resolve_batches(self.ramp)
        
        sampler = ECABatchSampler(
            num_samples=num_samples,
            batch_size=self.batch_size,
            k_values=list(self.k_values),
            seed=sampler_seed,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            k_max_start=self.k_max_start,
            warmup_batches=warmup_batches,
            ramp_batches=ramp_batches,
        )
        
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# ---------------------------
# Dataset
# ---------------------------
class ECADataset1D(Dataset):
    """
    On-the-fly ECA dataset.

    __getitem__ requires tuple indices with k:
      - (idx, k): sample rule from rules, use specified k
      - (idx, k, rule): use specified k and rule

    Use with ECADataConfig.create_dataloader() for proper usage.
    """

    def __init__(
        self,
        rules: Sequence[int],
        L: int,
        p: PType = 0.5,
        boundary: str = "wrap",
        seed: int = 0,
        num_samples: int = 1_000_000,
        dtype: torch.dtype = torch.long,
    ):
        # Store params
        self.rules = list(rules)
        self.L = L
        self.p = p
        self.boundary = boundary
        self.seed = seed
        self.num_samples = num_samples
        self.dtype = dtype
        
        # Validate
        self._validate()

        # Precompute rule LUTs (8-bit lookup per rule) for speed
        # lut[rule_idx] is tensor shape [8] with values {0,1} uint8
        self._rule_to_lut: Dict[int, torch.Tensor] = {r: self._make_rule_lut(r) for r in self.rules}

        # A stable 32-bit hash of seed (to mix with index deterministically)
        self._seed32 = self._stable_u32_from_int(self.seed)

    def _validate(self) -> None:
        if len(self.rules) == 0:
            raise ValueError("rules must be a non-empty sequence of ints in [0, 255].")
        for r in self.rules:
            if not isinstance(r, int) or r < 0 or r > 255:
                raise ValueError(f"Invalid rule: {r}. Must be int in [0, 255].")

        if self.L <= 0:
            raise ValueError("L must be > 0.")
        if self.boundary not in ("wrap", "zero"):
            raise ValueError('boundary must be "wrap" or "zero".')

        if isinstance(self.p, tuple):
            if len(self.p) != 2:
                raise ValueError("p as tuple must be (p_min, p_max).")
            pmin, pmax = self.p
            if not (0.0 <= pmin <= pmax <= 1.0):
                raise ValueError("p tuple must satisfy 0 <= p_min <= p_max <= 1.")
        else:
            if not (0.0 <= float(self.p) <= 1.0):
                raise ValueError("p must be in [0, 1].")

        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: Tuple[Any, ...]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        idx, k, rule_override = self._parse_index(index)

        # Derive a deterministic per-sample RNG seed from (seed, idx)
        sample_seed = self._mix_u32(self._seed32, idx)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(sample_seed))

        # Choose rule (unless overridden)
        rule = int(rule_override) if rule_override is not None else self._sample_from_list(self.rules, g)
        p = self._sample_p(self.p, g)

        # Sample initial condition x0 as uint8 {0,1} on CPU
        x0_u8 = self._bernoulli_u8(self.L, p, g)  # [L] uint8

        # Evolve k steps
        lut = self._rule_to_lut[rule]
        xk_u8 = self._evolve_k_steps(x0_u8, lut, k, boundary=self.boundary)

        # Convert to requested dtype (default long for embedding indexing)
        x0 = x0_u8.to(dtype=self.dtype)
        xk = xk_u8.to(dtype=self.dtype)

        meta = {"rule": rule, "k": k, "index": int(idx), "p": float(p)}
        return x0, xk, meta

    # ---------------------------
    # Index parsing
    # ---------------------------
    @staticmethod
    def _parse_index(index: Tuple[Any, ...]) -> Tuple[int, int, Optional[int]]:
        if not isinstance(index, tuple):
            raise TypeError(
                "Index must be (idx, k) or (idx, k, rule). "
                "Use ECADataConfig.create_dataloader() for proper usage. "
                f"Got: {type(index)} / {index}"
            )
        if len(index) == 2:
            idx, k = index
            return int(idx), int(k), None
        if len(index) == 3:
            idx, k, rule = index
            return int(idx), int(k), int(rule)
        raise TypeError(
            "Index must be (idx, k) or (idx, k, rule). "
            f"Got tuple of length {len(index)}: {index}"
        )

    # ---------------------------
    # Sampling helpers (deterministic via torch.Generator)
    # ---------------------------
    @staticmethod
    def _sample_from_list(values: Sequence[int], g: torch.Generator) -> int:
        # Uniform categorical
        j = int(torch.randint(low=0, high=len(values), size=(1,), generator=g).item())
        return int(values[j])

    @staticmethod
    def _sample_p(p: PType, g: torch.Generator) -> float:
        if isinstance(p, tuple):
            pmin, pmax = float(p[0]), float(p[1])
            u = float(torch.rand((), generator=g).item())
            return pmin + (pmax - pmin) * u
        return float(p)

    @staticmethod
    def _bernoulli_u8(L: int, p: float, g: torch.Generator) -> torch.Tensor:
        # uint8 {0,1}
        # torch.rand is fast; compare to p
        x = (torch.rand((L,), generator=g) < p).to(torch.uint8)
        return x

    # ---------------------------
    # Rule LUT + evolution
    # ---------------------------
    @staticmethod
    def _make_rule_lut(rule: int) -> torch.Tensor:
        # lut[i] = output for neighborhood index i (i in 0..7)
        bits = [(rule >> i) & 1 for i in range(8)]
        return torch.tensor(bits, dtype=torch.uint8)

    @staticmethod
    def _evolve_one_step(x_u8: torch.Tensor, lut_u8: torch.Tensor, boundary: str) -> torch.Tensor:
        # x_u8: [L] uint8 {0,1}
        left = torch.roll(x_u8, shifts=1, dims=0)
        right = torch.roll(x_u8, shifts=-1, dims=0)

        if boundary == "zero":
            left[0] = 0
            right[-1] = 0
        elif boundary != "wrap":
            raise ValueError(f"Unsupported boundary: {boundary}")

        # neighborhood index: 4*left + 2*center + right in [0..7]
        idx = ((left << 2) | (x_u8 << 1) | right).long()
        return lut_u8[idx]

    @classmethod
    def _evolve_k_steps(cls, x0_u8: torch.Tensor, lut_u8: torch.Tensor, k: int, boundary: str) -> torch.Tensor:
        x = x0_u8
        for _ in range(k):
            x = cls._evolve_one_step(x, lut_u8, boundary)
        return x

    # ---------------------------
    # Stable deterministic hashing / mixing for per-index seeds
    # ---------------------------
    @staticmethod
    def _stable_u32_from_int(x: int) -> int:
        # stable 32-bit from integer using SHA256
        b = str(int(x)).encode("utf-8")
        h = hashlib.sha256(b).digest()
        return int.from_bytes(h[:4], "little", signed=False)

    @staticmethod
    def _mix_u32(seed32: int, idx: int) -> int:
        # simple 32-bit mix (SplitMix-like constants)
        x = (seed32 + (idx & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
        x ^= (x >> 16)
        x = (x * 0x85EBCA6B) & 0xFFFFFFFF
        x ^= (x >> 13)
        x = (x * 0xC2B2AE35) & 0xFFFFFFFF
        x ^= (x >> 16)
        return x


# ---------------------------
# Batch Sampler with Curriculum Support
# ---------------------------
class ECABatchSampler(Sampler[List[Tuple[int, int]]]):
    """
    Batch sampler that emits (idx, k) tuples ensuring same-k per batch.
    Supports optional curriculum learning.

    Parameters:
      num_samples: virtual dataset length (use len(dataset))
      batch_size:  batch size
      k_values:    all possible k values to sample from
      seed:        sampler RNG seed (controls batch k selection + index order)
      drop_last:   drop incomplete final batch
      shuffle:     shuffle indices each epoch
      k_max_start: starting max k for curriculum (if None, no curriculum)
      warmup_batches: batches before curriculum starts ramping
      ramp_batches: batches to ramp from k_max_start to max(k_values)
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        k_values: Sequence[int],
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
        k_max_start: Optional[int] = None,
        warmup_batches: int = 0,
        ramp_batches: int = 0,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(k_values) == 0:
            raise ValueError("k_values must be non-empty")
        
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.k_values = sorted([int(k) for k in k_values])  # sort for curriculum
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        
        # Curriculum params
        self.warmup_batches = int(warmup_batches)
        self.ramp_batches = int(ramp_batches)
        
        # Precompute k bounds
        self.k_min = min(self.k_values)
        self.k_max = max(self.k_values)
        
        # Default k_max_start to k_min if not provided
        if k_max_start is None:
            self.k_max_start = self.k_min
        else:
            self.k_max_start = int(k_max_start)
            # Validate k_max_start if explicitly provided
            if self.k_max_start < self.k_min:
                raise ValueError(f"k_max_start ({self.k_max_start}) must be >= min(k_values) ({self.k_min})")
            if self.k_max_start > self.k_max:
                raise ValueError(f"k_max_start ({self.k_max_start}) must be <= max(k_values) ({self.k_max})")
        
        # Epoch counter for curriculum tracking and shuffle seeding
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for curriculum tracking and shuffle seeding.
        
        Call this before each epoch for:
        - Reproducible shuffling in distributed training
        - Easy checkpoint resume
        
        If not called, the sampler auto-increments epoch at end of each iteration.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        self._epoch = int(epoch)

    def _get_global_batch_idx(self, batch_within_epoch: int) -> int:
        """Compute global batch index from epoch and batch within epoch."""
        return self._epoch * len(self) + batch_within_epoch

    def _get_k_values_for_batch(self, global_batch_idx: int) -> List[int]:
        """Get available k values for a given global batch index based on curriculum."""
        # No curriculum: use all k_values immediately
        if self.ramp_batches == 0:
            return self.k_values
        
        # Before warmup: use k <= k_max_start
        if global_batch_idx < self.warmup_batches:
            return [k for k in self.k_values if k <= self.k_max_start]
        
        # After ramp: use all k_values
        progress_batch = global_batch_idx - self.warmup_batches
        if progress_batch >= self.ramp_batches:
            return self.k_values
        
        # During ramp: linearly interpolate current_k_max
        t = progress_batch / self.ramp_batches  # 0.0 to 1.0
        current_k_max = self.k_max_start + t * (self.k_max - self.k_max_start)
        
        # Filter k_values to those <= current_k_max
        return [k for k in self.k_values if k <= current_k_max]

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        # Deterministic permutation of indices (seeded by epoch for reproducible shuffle)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self._epoch)

        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            order = list(range(self.num_samples))

        batch_within_epoch = 0
        for start in range(0, self.num_samples, self.batch_size):
            end = start + self.batch_size
            if end > self.num_samples and self.drop_last:
                break

            # Get k values based on curriculum
            global_batch_idx = self._get_global_batch_idx(batch_within_epoch)
            k_values = self._get_k_values_for_batch(global_batch_idx)

            # Uniform k from current k_values
            ki = int(torch.randint(low=0, high=len(k_values), size=(1,), generator=g).item())
            k = int(k_values[ki])

            indices = order[start:min(end, self.num_samples)]
            yield [(int(i), k) for i in indices]
            batch_within_epoch += 1
        
        # Auto-increment epoch at end of iteration (for users who don't call set_epoch)
        self._epoch += 1

    def __len__(self) -> int:
        n = self.num_samples // self.batch_size
        if not self.drop_last and (self.num_samples % self.batch_size != 0):
            n += 1
        return n


# ---------------------------
# Quick self-test / example
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ECA Dataset Examples")
    print("=" * 60)
    
    # Example 1: Basic usage
    print("\n1. Basic usage (same-k per batch):")
    config = ECADataConfig(
        rules=[30, 110],
        L=64,
        p=0.5,
        batch_size=8,
        k_values=list(range(1, 9)),
        total_batches=1000,
    )
    loader = config.create_dataloader()
    x0, xk, meta = next(iter(loader))
    print(f"   Shapes: x0={x0.shape}, xk={xk.shape}")
    print(f"   dtypes: x0={x0.dtype}, xk={xk.dtype}")
    print(f"   Batch k values: {meta['k'].tolist()} (all same)")
    print(f"   Total batches: {len(loader)}")
    
    # Example 2: Curriculum learning with absolute batches
    print("\n2. Curriculum learning (absolute batches):")
    config_curriculum = ECADataConfig(
        rules=[30, 110],
        L=64,
        batch_size=8,
        k_values=list(range(1, 9)),
        k_max_start=2,
        ramp=100,  # ramp over 100 batches
        total_batches=1000,
    )
    sampler = config_curriculum.create_dataloader().batch_sampler
    print(f"   Batch 0 k_values:   {sampler._get_k_values_for_batch(0)}")
    print(f"   Batch 50 k_values:  {sampler._get_k_values_for_batch(50)}")
    print(f"   Batch 100 k_values: {sampler._get_k_values_for_batch(100)}")
    
    # Example 3: Curriculum learning with fractional ramp
    print("\n3. Curriculum learning (fractional ramp = 10% of total):")
    config_frac = ECADataConfig(
        rules=[30],
        L=64,
        batch_size=8,
        k_values=list(range(1, 9)),
        ramp=0.1,  # 10% of total_batches = 100 batches
        total_batches=1000,
    )
    sampler_frac = config_frac.create_dataloader().batch_sampler
    print(f"   ramp_batches resolved to: {sampler_frac.ramp_batches}")
    print(f"   Batch 0 k_values:   {sampler_frac._get_k_values_for_batch(0)}")
    print(f"   Batch 50 k_values:  {sampler_frac._get_k_values_for_batch(50)}")
    print(f"   Batch 100 k_values: {sampler_frac._get_k_values_for_batch(100)}")
    
    # Example 4: Multi-worker test
    print("\n4. Multi-worker test (num_workers=2):")
    config_workers = ECADataConfig(
        rules=[30],
        L=64,
        batch_size=8,
        k_values=list(range(1, 5)),
        total_batches=20,
    )
    loader_workers = config_workers.create_dataloader(num_workers=2)
    batch_count = sum(1 for _ in loader_workers)
    print(f"   Processed {batch_count} batches with 2 workers ✓")
    
    # Example 5: Verify determinism
    print("\n5. Determinism check:")
    config1 = ECADataConfig(rules=[30], L=64, batch_size=8, k_values=[1, 2, 3], total_batches=100)
    config2 = ECADataConfig(rules=[30], L=64, batch_size=8, k_values=[1, 2, 3], total_batches=100)
    loader1 = config1.create_dataloader()
    loader2 = config2.create_dataloader()
    x0_1, xk_1, meta_1 = next(iter(loader1))
    x0_2, xk_2, meta_2 = next(iter(loader2))
    print(f"   Same x0: {torch.equal(x0_1, x0_2)}")
    print(f"   Same xk: {torch.equal(xk_1, xk_2)}")
    print(f"   Same k:  {torch.equal(meta_1['k'], meta_2['k'])}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
