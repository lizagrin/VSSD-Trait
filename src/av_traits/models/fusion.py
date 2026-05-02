"""Trait-wise late fusion + auxiliary classification head"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..config import cfg


class TraitWiseLateFusion(nn.Module):
    """Per-trait soft fusion of an arbitrary subset of modalities"""
    def __init__(self, hidden_dim: int, num_traits: int, modalities: List[str]):
        super().__init__()
        self.modalities = list(modalities)
        self.heads = nn.ModuleDict({m: nn.Linear(hidden_dim, num_traits)
                                    for m in self.modalities})
        if cfg.fusion_mode == "weighted":
            self.gate = nn.Linear(hidden_dim * len(self.modalities),
                                  num_traits * len(self.modalities))
            self.cat_head = None
        elif cfg.fusion_mode == "concat":
            self.gate = None
            self.cat_head = nn.Linear(hidden_dim * len(self.modalities), num_traits)
        else:
            raise ValueError(f"Unknown fusion mode: {cfg.fusion_mode!r}")

    def forward(self, feats: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        ys = {m: torch.sigmoid(self.heads[m](feats[m])) for m in self.modalities}
        cat = torch.cat([feats[m] for m in self.modalities], dim=-1)
        B = cat.shape[0]
        if cfg.fusion_mode == "weighted":
            g = self.gate(cat).view(B, -1, len(self.modalities))
            alpha = torch.softmax(g, dim=-1)
            stacked = torch.stack([ys[m] for m in self.modalities], dim=-1)  # [B, T, M]
            y = (alpha * stacked).sum(-1)
        else:
            y = torch.sigmoid(self.cat_head(cat))
            alpha = torch.full((B, y.shape[1], len(self.modalities)),
                               1.0 / len(self.modalities), device=cat.device)
        return y, ys, alpha


class BinHead(nn.Module):
    """Per-trait 3-class head used by the auxiliary classification loss."""
    def __init__(self, in_dim: int, num_traits: int = 5, num_bins: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_traits * num_bins)
        self.num_traits = num_traits
        self.num_bins = num_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).view(-1, self.num_traits, self.num_bins)
