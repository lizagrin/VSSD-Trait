"""I/O helpers."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..config import cfg


def save_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, encoding="utf-8")
    return path


def load_csv(path: str, index_col: int = 0) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col, encoding="utf-8")


def save_predictions_csv(video_ids: Iterable[str], preds: np.ndarray,
                         out_path: str) -> pd.DataFrame:
    df = pd.DataFrame(preds, columns=list(cfg.trait_names))
    df.insert(0, "video_id", list(video_ids))
    df.to_csv(out_path, index=False)
    return df


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["model_state_dict"]
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        own = model.state_dict()
        compat = {k: v for k, v in sd.items()
                  if k in own and v.shape == own[k].shape}
        model.load_state_dict(compat, strict=False)
        print(f"Loaded {len(compat)}/{len(sd)} parameters non-strictly.")
    return obj
