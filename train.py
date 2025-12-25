"""
Morpher Training Script

This script trains Morpher models on Sudoku/ARC problems using HRM's training
infrastructure. It bridges the main Morpher repo with the HRM submodule.

Usage:
    python train.py --config config/morpher_act.yaml
    python train.py --config config/morpher_fixed.yaml
    
    # Or with Hydra overrides:
    python train.py --config config/morpher_act.yaml epochs=1000 global_batch_size=128
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add paths for imports
REPO_ROOT = Path(__file__).parent


from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

# HRM imports
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from models.losses import ACTLossHead

# Morpher imports
from morpher_wrapper import MorpherACT, MorpherFixedSteps, MorpherWrapperConfig

# Optimizer imports (from HRM)
try:
    from adam_atan2 import AdamATan2
except ImportError:
    from torch.optim import AdamW as AdamATan2
    print("Warning: adam_atan2 not found, using AdamW")

try:
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
except ImportError:
    CastedSparseEmbeddingSignSGD_Distributed = None


# =============================================================================
# Config classes (similar to HRM's pretrain.py)
# =============================================================================
class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class TrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_path: str
    
    global_batch_size: int
    epochs: int
    
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    
    weight_decay: float
    beta1: float
    beta2: float
    
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


# =============================================================================
# Model factory
# =============================================================================
MODEL_CLASSES = {
    "MorpherACT": MorpherACT,
    "MorpherFixedSteps": MorpherFixedSteps,
}

LOSS_CLASSES = {
    "ACTLossHead": ACTLossHead,
}


def create_model(config: TrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    """Create Morpher model with loss head."""
    
    model_cfg = MorpherWrapperConfig(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )
    
    # Get model and loss classes
    model_cls = MODEL_CLASSES.get(config.arch.name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {config.arch.name}. Available: {list(MODEL_CLASSES.keys())}")
    
    loss_cls = LOSS_CLASSES.get(config.arch.loss.name)
    if loss_cls is None:
        raise ValueError(f"Unknown loss: {config.arch.loss.name}")
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore
        
        # Broadcast params in distributed mode
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
    
    # Setup optimizers
    optimizers = []
    optimizer_lrs = []
    
    # Puzzle embedding optimizer (if available and sparse embedding is supported)
    if hasattr(model.model, 'puzzle_emb') and model.model.puzzle_emb is not None:
        if CastedSparseEmbeddingSignSGD_Distributed is not None:
            try:
                optimizers.append(
                    CastedSparseEmbeddingSignSGD_Distributed(
                        model.model.puzzle_emb.parameters(),
                        lr=0,
                        weight_decay=config.puzzle_emb_weight_decay,
                        world_size=world_size
                    )
                )
                optimizer_lrs.append(config.puzzle_emb_lr)
            except Exception:
                pass  # Skip if puzzle_emb doesn't support this optimizer
    
    # Main optimizer
    optimizers.append(
        AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    )
    optimizer_lrs.append(config.lr)
    
    return model, optimizers, optimizer_lrs


# =============================================================================
# Data loading (reusing HRM's PuzzleDataset)
# =============================================================================
def create_dataloader(config: TrainConfig, split: str, rank: int, world_size: int, **kwargs):
    """Create dataloader for puzzle dataset."""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


# =============================================================================
# Training utilities
# =============================================================================
def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, 
    num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def compute_lr(base_lr: float, config: TrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def init_train_state(config: TrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)
    
    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: TrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def train_batch(config: TrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return
    
    batch = {k: v.cuda() for k, v in batch.items()}
    
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)
    
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    
    ((1 / global_batch_size) * loss).backward()
    
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()
    
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        
        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: TrainConfig, train_state: TrainState, eval_loader: DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        all_preds = {}
        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
            
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                if all_finish:
                    break
            
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())
            
            del carry, preds, batch, all_finish
            
            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size
        
        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))
        
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}
                
                return reduced_metrics


def save_code_and_config(config: TrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Copy source files
    for src_file in ["src/morpher.py", "src/morpher_wrapper.py", "src/utils.py"]:
        src_path = REPO_ROOT / src_file
        if src_path.exists():
            shutil.copy(src_path, os.path.join(config.checkpoint_path, os.path.basename(src_file)))
    
    # Dump config
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)
    
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> TrainConfig:
    objects = [None]
    if rank == 0:
        config = TrainConfig(**hydra_config)
        
        if config.project_name is None:
            config.project_name = f"Morpher {os.path.basename(config.data_path).capitalize()}"
        if config.run_name is None:
            config.run_name = f"{config.arch.name} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        
        objects = [config]
    
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    
    return objects[0]  # type: ignore


# =============================================================================
# Main entry point
# =============================================================================
@hydra.main(config_path="config", config_name="morpher_act", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)
    
    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."
    
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    
    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    
    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    
    # Training loop
    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")
        
        # Train
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
        
        # Evaluate
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
        
        # Checkpoint
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
