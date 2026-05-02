"""Learning-rate schedule utilities (warmup → cosine decay)."""
from __future__ import annotations

import math

import torch


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int,
                   warmup_ratio: float = 0.08
                   ) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_ratio`` of training, then half-cosine decay."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
