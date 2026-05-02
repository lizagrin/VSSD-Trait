"""Per-stage train / validation loop."""
from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import cfg
from .losses import compute_loss
from .metrics import metrics_np
from .optim import make_optimizer, set_backbone_train_mode
from .schedules import make_scheduler


def _to_dev(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    train: bool = True,
    device: Optional[str] = None,
) -> dict:
    """One pass over the loader; returns the validation/training metrics."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train(train)
    ps, ys = [], []
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = _to_dev(batch, device)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=cfg.amp):
            out = model(batch)
            loss = compute_loss(out, batch["target"])
        if train:
            ls = loss / cfg.grad_accum_steps
            (scaler.scale(ls).backward() if scaler else ls.backward())
            if (step + 1) % cfg.grad_accum_steps == 0 or step + 1 == len(loader):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
        ps.append(out["pred"].detach().cpu().numpy())
        ys.append(batch["target"].detach().cpu().numpy())
    return metrics_np(np.concatenate(ps), np.concatenate(ys))


def fit_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
    epochs: int,
    lr_heads: float,
    lr_backbone: float,
    backbone_mode: str,
) -> str:
    """Train ``model`` for ``epochs`` epochs and save the best checkpoint."""
    set_backbone_train_mode(model, backbone_mode)
    opt = make_optimizer(model, lr_heads, lr_backbone)
    sps = math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))
    sch = make_scheduler(opt, max(1, epochs * sps))
    sca = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    best, best_path = -1e9, os.path.join(cfg.checkpoints_dir, f"best_{name}.pt")

    for e in range(1, epochs + 1):
        print(f"\n[{name}] epoch {e}/{epochs}")
        run_epoch(model, train_loader, opt, sch, sca, train=True)
        m = run_epoch(model, val_loader, None, None, None, train=False)
        score = m["ccc_mean"] + 0.25 * m["macc_mean"]
        print({k: round(v, 4) for k, v in m.items()
               if k in ("macc_mean", "ccc_mean", "mae_mean")})
        if score > best:
            best = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "metrics": m,
                "cfg": asdict(cfg),
                "epoch": e,
                "stage_name": name,
            }, best_path)
            print("saved", best_path)
    return best_path
