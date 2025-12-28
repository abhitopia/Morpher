# eca/tasks/forward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from .base import Batch, Task


@dataclass(frozen=True)
class ForwardTaskConfig:
    condition_on_rule: bool = True
    condition_on_k: bool = True
    loss: str = "ce"  # "ce" (recommended with logits [B,L,2]) or "bce" (logits [B,L])


class ForwardPredictTask(Task):
    """
    Task A: Given (x0, rule, k) predict xk.
    """

    def __init__(self, cfg: ForwardTaskConfig):
        self.cfg = cfg

    def collate(self, raw: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Batch:
        x0s, xks, metas = zip(*raw)
        x0 = torch.stack(x0s, dim=0)  # [B,L]
        xk = torch.stack(xks, dim=0)  # [B,L]

        rule = torch.tensor([m["rule"] for m in metas], dtype=torch.long)
        k = torch.tensor([m["k"] for m in metas], dtype=torch.long)

        return Batch(x0=x0, xk=xk, rule=rule, k=k, meta={"raw_meta": metas})

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        """
        Expected outputs:
          - outputs["xk_logits"] is either [B,L,2] for CE or [B,L] for BCE
        """
        logits = outputs["xk_logits"]
        target = batch.xk

        if self.cfg.loss == "ce":
            # logits: [B,L,2], target: [B,L] with values 0/1
            return F.cross_entropy(logits.view(-1, 2), target.view(-1))
        elif self.cfg.loss == "bce":
            # logits: [B,L], target: [B,L] {0,1}
            return F.binary_cross_entropy_with_logits(logits, target.float())
        else:
            raise ValueError(f"Unknown loss: {self.cfg.loss}")

    @torch.no_grad()
    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Batch) -> Dict[str, torch.Tensor]:
        logits = outputs["xk_logits"]
        target = batch.xk

        if self.cfg.loss == "ce":
            pred = logits.argmax(dim=-1)  # [B,L]
        else:
            pred = (torch.sigmoid(logits) > 0.5).long()

        bit_acc = (pred == target).float().mean()

        # Exact match (all bits correct per sample) is a good "hard" metric
        exact = (pred == target).all(dim=-1).float().mean()

        # Mean normalized Hamming distance
        ham = (pred != target).float().mean()

        return {
            "bit_acc": bit_acc,
            "exact_match": exact,
            "hamming": ham,
        }
