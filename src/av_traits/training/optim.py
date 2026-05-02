"""Optimizer + parameter-group helpers for staged unfreezing."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..config import cfg


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def set_backbone_train_mode(model: nn.Module, mode: str = "frozen") -> None:
    """Configure which parts of the visual backbone receive gradients"""
    if not hasattr(model, "visual"):
        return
    freeze_module(model.visual.backbone)
    if mode == "last_stage" and hasattr(model.visual.backbone, "layers"):
        unfreeze_module(model.visual.backbone.layers[-1])
        if hasattr(model.visual.backbone, "norm"):
            unfreeze_module(model.visual.backbone.norm)
    elif mode == "all":
        unfreeze_module(model.visual.backbone)
    elif mode == "frozen":
        pass
    else:
        raise ValueError(f"Unknown backbone mode: {mode!r}")


def make_optimizer(model: nn.Module, lr_heads: float, lr_backbone: float
                   ) -> torch.optim.Optimizer:
    """AdamW with separate parameter groups for backbone and heads."""
    backbone_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("visual.backbone") else head_params).append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_heads,
                       "weight_decay": cfg.weight_decay})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone,
                       "weight_decay": cfg.weight_decay})
    return torch.optim.AdamW(groups)
