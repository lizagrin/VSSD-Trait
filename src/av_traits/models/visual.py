"""Visual temporal module"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import cfg


class AttentionPool(nn.Module):
    """Soft attention pool over the second axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        l = self.score(x).squeeze(-1)
        if mask is not None:
            l = l.float().masked_fill(~mask.bool(), torch.finfo(l.dtype).min)
        a = torch.softmax(l.float(), dim=-1).to(x.dtype)
        return torch.einsum("bn,bnd->bd", a, x), a


class MeanPool(nn.Module):
    """Mean pooling alternative used in the architectural ablation """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            return x.mean(dim=1), torch.full(x.shape[:2], 1.0 / x.shape[1], device=x.device)
        m = mask.float()
        s = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        return s, m / m.sum(1, keepdim=True).clamp_min(1)


def make_pool(dim: int) -> nn.Module:
    """Pick the pooling module per the active config"""
    return AttentionPool(dim) if cfg.pooling_mode == "attention" else MeanPool()


class VisualTemporalModule(nn.Module):
    """Backbone-agnostic spatio-temporal aggregator"""

    def __init__(self, backbone: nn.Module, visual_dim: int,
                 hidden_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.backbone = backbone
        self.frame_norm = nn.LayerNorm(visual_dim)
        self.frame_pool = make_pool(visual_dim)
        self.window_proj = nn.Sequential(
            nn.Linear(visual_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                               dropout=dropout, batch_first=True)
        self.temporal_ln = nn.LayerNorm(hidden_dim)
        self.window_pool = make_pool(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, faces: torch.Tensor, face_scores: torch.Tensor):
        B, W, K, C, H, Wi = faces.shape
        feats = self.backbone.forward_features(faces.view(B * W * K, C, H, Wi))
        D = feats.shape[-1]
        feats = self.frame_norm(feats.view(B, W, K, D))

        if not cfg.use_windowing:
            # clip-level baseline (Table 2): collapse all frames at once
            clip_token = feats.mean(dim=(1, 2))
            score = face_scores.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
            clip_vec = self.out(self.window_proj(torch.cat([clip_token, score], -1)))
            return clip_vec, {
                "frame_attn": None,
                "window_attn": torch.ones(B, 1, device=feats.device),
                "frame_feats": feats,
            }

        ft = feats.view(B * W, K, D)
        pooled, fa = self.frame_pool(ft)
        pooled = pooled.view(B, W, D)
        fa = fa.view(B, W, K) if fa.ndim == 2 else fa
        win = self.window_proj(torch.cat([pooled, face_scores.mean(-1, keepdim=True)], -1))
        t, _ = self.bigru(win)
        a, _ = self.self_attn(t, t, t, need_weights=False)
        t = self.temporal_ln(t + a)
        clip_vec, wa = self.window_pool(t)
        return self.out(clip_vec), {
            "frame_attn": fa,
            "window_attn": wa,
            "frame_feats": feats,
        }
