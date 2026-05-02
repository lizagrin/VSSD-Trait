"""Big Five personality metrics: per-trait MAE, mACC and CCC."""
from __future__ import annotations

import numpy as np
import torch

from ..config import cfg


def t_ccc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-trait CCC, computed in pure torch (differentiable)"""
    mx = pred.mean(0); my = target.mean(0)
    vx = pred.var(0, unbiased=False); vy = target.var(0, unbiased=False)
    cov = ((pred - mx) * (target - my)).mean(0)
    return (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)


# Backwards-compatible aliases used by downstream scripts.
tensor_ccc = t_ccc


def mean_ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return t_ccc(pred, target).mean()


def metrics_np(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute the standard FIv2 metrics from numpy arrays."""
    mae = np.mean(np.abs(preds - targets), axis=0)
    macc = 1.0 - mae
    cccs = []
    for i in range(targets.shape[1]):
        x, y = preds[:, i], targets[:, i]
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mx) * (y - my)).mean()
        cccs.append((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8))

    out = {
        "mae_mean":  float(mae.mean()),
        "macc_mean": float(macc.mean()),
        "ccc_mean":  float(np.mean(cccs)),
    }
    for i, n in enumerate(cfg.trait_names):
        out[f"mae_{n}"]  = float(mae[i])
        out[f"macc_{n}"] = float(macc[i])
        out[f"ccc_{n}"]  = float(cccs[i])
    return out
