"""Audio module: log-mel CNN + hand-crafted prosody / eGeMAPS branch."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import cfg


class AudioDeepCNN(nn.Module):
    """Compact 2-D CNN over log-mel spectrogram - fixed-size embedding."""

    def __init__(self, out_dim: int = 256, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(128, out_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel.unsqueeze(1))


class _HCBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioModule(nn.Module):
    """Two-branch audio encoder"""

    def __init__(self, prosody_dim: int, ege_dim: int,
                 hidden_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.deep = AudioDeepCNN(out_dim=cfg.audio_deep_dim, p=dropout)
        self.hc = _HCBranch(prosody_dim + ege_dim,
                            hidden_dim=cfg.audio_hc_dim,
                            out_dim=cfg.audio_hc_dim,
                            p=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.audio_deep_dim + cfg.audio_hc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, mel: torch.Tensor, prosody: torch.Tensor, ege: torch.Tensor):
        d = self.deep(mel)
        h = self.hc(torch.cat([prosody, ege], dim=-1))
        return self.fuse(torch.cat([d, h], dim=-1))
