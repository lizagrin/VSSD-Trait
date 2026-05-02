"""Composite loss used by the multi-task training objective."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import cfg
from .metrics import t_ccc as _t_ccc

BIN_BOUNDS = [0.0, 0.4, 0.6, 1.000001]
_huber = nn.SmoothL1Loss(beta=cfg.huber_beta)
_ce = nn.CrossEntropyLoss()


def target_to_bins(y: torch.Tensor) -> torch.Tensor:
    borders = torch.tensor(BIN_BOUNDS[1:-1], device=y.device)
    return torch.bucketize(y, boundaries=borders).long()


def pairwise_loss(emb: torch.Tensor, y: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    """Trait-conditional metric-learning loss on the joint clip embedding."""
    B = emb.shape[0]
    if B < 2:
        return emb.new_tensor(0.0)
    bins = target_to_bins(y)
    dist = 1 - F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    losses = []
    eye = torch.eye(B, device=emb.device)
    for t in range(y.shape[1]):
        same = (bins[:, t].unsqueeze(1) == bins[:, t].unsqueeze(0)).float() * (1.0 - eye)
        diff = (1.0 - same) * (1.0 - eye)
        pos = (dist * same).sum() / same.sum().clamp_min(1.0)
        neg = (F.relu(margin - dist) * diff).sum() / diff.sum().clamp_min(1.0)
        losses.append(pos + neg)
    return torch.stack(losses).mean()


def compute_loss(outputs: dict, target: torch.Tensor) -> torch.Tensor:
    """Composite training loss"""
    pred = outputs["pred"]
    loss = _huber(pred, target)

    if cfg.use_aux_heads:
        for m, py in outputs["per_mod"].items():
            loss = loss + cfg.lambda_aux_modal * _huber(py, target)

    if cfg.use_ccc_loss:
        loss = loss + cfg.lambda_ccc * (1.0 - _t_ccc(pred, target).mean())

    bins = target_to_bins(target)
    bin_l = 0.0
    for t in range(cfg.num_traits):
        bin_l = bin_l + _ce(outputs["bin_logits"][:, t, :], bins[:, t])
    loss = loss + cfg.lambda_bin * bin_l / cfg.num_traits

    if "feats" in outputs and len(outputs["feats"]) >= 2:
        emb = torch.cat(list(outputs["feats"].values()), dim=-1)
        loss = loss + cfg.lambda_pair * pairwise_loss(emb, target)

    return loss
