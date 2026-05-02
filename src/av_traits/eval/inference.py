"""Inference and per-modality evaluation."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import cfg
from ..training.metrics import metrics_np


def _to_dev(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate_modalities(model: torch.nn.Module, loader: DataLoader,
                        modalities: Sequence[str]) -> dict:
    """Evaluate the model with only the requested modalities active"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    saved = list(model.modalities)
    model.modalities = list(modalities)
    try:
        ps, ys = [], []
        for batch in tqdm(loader, total=len(loader), desc="+".join(modalities)):
            batch = _to_dev(batch, device)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out = model(batch)
            ps.append(out["pred"].detach().cpu().numpy())
            ys.append(batch["target"].detach().cpu().numpy())
        return metrics_np(np.concatenate(ps), np.concatenate(ys))
    finally:
        model.modalities = saved


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader
                   ) -> Tuple[list, np.ndarray]:
    """Run the model on the entire loader and return ``(video_ids, preds)``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    ids, ps = [], []
    for batch in tqdm(loader, total=len(loader)):
        ids.extend(batch["video_id"])
        batch = _to_dev(batch, device)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            out = model(batch)
        ps.append(out["pred"].detach().cpu().numpy())
    return ids, np.concatenate(ps, axis=0)
