"""Reproduce temporal saliency of clip windows"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.utils.io import load_checkpoint, save_csv


def _to_dev(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def compute_window_attention(model, loader,
                             n_windows: int = 14,
                             clip_seconds: float = 15.0) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    accum = np.zeros(n_windows, dtype=np.float64); n = 0
    for batch in tqdm(loader, total=len(loader), desc="window α"):
        batch = _to_dev(batch, device)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            out = model(batch)
        wa = out.get("visual", {}).get("window_attn")
        if wa is None:
            continue
        wa = wa.float().cpu().numpy()
        if wa.shape[1] != n_windows:
            continue
        accum += wa.sum(0)
        n += wa.shape[0]

    alpha = accum / max(n, 1)
    max_a = alpha.max() if alpha.max() > 0 else 1.0
    win_sec = clip_seconds / n_windows

    rows = []
    for w in range(n_windows):
        t0 = w * win_sec; t1 = t0 + 2.0
        rows.append({
            "Окно": w + 1,
            "Время, с": f"{int(round(t0))}–{int(round(t1))}",
            "Средний вес α": round(float(alpha[w]), 3),
            "Доля от макс. (%)": int(round(100 * alpha[w] / max_a)),
        })
    return pd.DataFrame(rows).set_index("Окно")


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str,
                   default="results/tables/table_9_window_attention.csv")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, _, test_loader = build_loaders(cfg)
    model = build_model_for_cfg(cfg, train_loader)
    load_checkpoint(model, args.checkpoint)

    df = compute_window_attention(model, test_loader,
                                  n_windows=cfg.eval_windows,
                                  clip_seconds=15.0)
    print(df.to_string())
    save_csv(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
