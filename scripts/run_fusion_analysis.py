"""Reproduce modality fusion weights per Big Five trait."""
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

TRAITS_RU = {
    "openness":          "Openness (O)",
    "conscientiousness": "Conscientiousness (C)",
    "extraversion":      "Extraversion (E)",
    "agreeableness":     "Agreeableness (A)",
    "neuroticism":       "Neuroticism (N)",
}


def _to_dev(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def compute_fusion_weights(model, loader) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    accum = None; n = 0
    for batch in tqdm(loader, total=len(loader), desc="alpha"):
        batch = _to_dev(batch, device)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            out = model(batch)
        a = out["alpha"].float().sum(0).cpu().numpy()       # [T, M]
        accum = a if accum is None else accum + a
        n += out["alpha"].shape[0]

    alpha = accum / max(n, 1)
    rows = []
    for ti, tname in enumerate(cfg.trait_names):
        wv, wa, wt = float(alpha[ti, 0]), float(alpha[ti, 1]), float(alpha[ti, 2])
        dom = {0: "видео", 1: "аудио", 2: "текст"}[int(np.argmax([wv, wa, wt]))]
        if max(wv, wa, wt) - min(wv, wa, wt) < 0.06:
            dom = "сбалансированная"
        rows.append({
            "Черта": TRAITS_RU[tname],
            "w_V (видео)":  round(wv, 3),
            "w_A (аудио)":  round(wa, 3),
            "w_T (текст)":  round(wt, 3),
            "Доминирующая модальность": dom,
        })
    return pd.DataFrame(rows).set_index("Черта")


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str,
                   default="results/tables/table_8_fusion_weights.csv")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, _, test_loader = build_loaders(cfg)
    model = build_model_for_cfg(cfg, train_loader)
    load_checkpoint(model, args.checkpoint)

    df = compute_fusion_weights(model, test_loader)
    print(df.to_string())
    save_csv(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
