from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.eca import ECADataConfig, ECABatchSampler
from tasks.base import Task


class ECADataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ECA datasets.
    
    Uses ECADataConfig.create_dataloader() for clean setup.
    """
    
    def __init__(
        self,
        train_config: ECADataConfig,
        val_config: ECADataConfig,
        task: Task,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.task = task
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_loader = self.train_config.create_dataloader(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task.collate,
        )
        self._val_loader = self.val_config.create_dataloader(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task.collate,
        )

    @property
    def train_sampler(self) -> Optional[ECABatchSampler]:
        if self._train_loader is not None:
            return self._train_loader.batch_sampler
        return None

    @property
    def val_sampler(self) -> Optional[ECABatchSampler]:
        if self._val_loader is not None:
            return self._val_loader.batch_sampler
        return None

    def on_train_epoch_start(self) -> None:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.trainer.current_epoch)

    def on_validation_epoch_start(self) -> None:
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(self.trainer.current_epoch)

    def train_dataloader(self) -> DataLoader:
        assert self._train_loader is not None
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        assert self._val_loader is not None
        return self._val_loader
