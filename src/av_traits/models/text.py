"""Text branch — frozen SBERT + small projection MLP"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..config import cfg


class TextModule(nn.Module):
    """Maps a SBERT sentence embedding into the shared space"""

    def __init__(self, text_dim: int, hidden_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(text_emb)


_SBERT = None


def _ensure_sbert():
    """Lazy-load the SBERT encoder so importing the module is cheap"""
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SBERT = SentenceTransformer(cfg.sbert_model, device=device)
        for p in _SBERT.parameters():
            p.requires_grad = False
    return _SBERT


@torch.no_grad()
def encode_texts_with_sbert(texts: List[str]) -> torch.Tensor:
    """Encode a list of strings into normalised SBERT vectors"""
    enc = _ensure_sbert()
    embs = enc.encode(list(texts), convert_to_tensor=True,
                      normalize_embeddings=True, show_progress_bar=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return embs.to(device).float()
