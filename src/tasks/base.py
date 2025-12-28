from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

import torch


@dataclass
class Batch:
    x0: torch.Tensor          # [B, L] long (0/1)
    xk: torch.Tensor          # [B, L] long (0/1)
    rule: torch.Tensor        # [B] long
    k: torch.Tensor           # [B] long
    meta: Dict[str, Any]      # whatever else


class Task(Protocol):
    """
    A Task defines how raw dataset samples become a model batch, plus loss/metrics.
    """

    def collate(self, raw: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Batch:
        ...

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        ...

    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Batch) -> Dict[str, torch.Tensor]:
        ...
