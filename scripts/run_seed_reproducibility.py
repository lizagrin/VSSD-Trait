"""Reproduce 5-seed reproducibility study"""
from __future__ import annotations

import argparse
import gc
import math

import pandas as pd
import torch

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.eval import evaluate_modalities
from av_traits.training import fit_stage
from av_traits.utils.io import save_csv
from av_traits.utils.seeds import set_all_seeds

SEEDS = [42, 123, 456, 789, 2024]
T_4 = 2.776  # Student's t critical value for df=4 at 95% CI


def _train_one_seed(seed, train_loader, val_loader, test_loader):
    set_all_seeds(seed)
    cfg.active_modalities = ("V", "A", "T")
    m = build_model_for_cfg(cfg, train_loader)
    fit_stage(m, train_loader, val_loader, f"seed{seed}_s1",
              cfg.stage1_epochs, cfg.lr_heads_stage1, 0.0, "frozen")
    fit_stage(m, train_loader, val_loader, f"seed{seed}_s2",
              cfg.stage2_epochs, cfg.lr_heads_stage2, cfg.lr_backbone_stage2,
              "last_stage")
    fit_stage(m, train_loader, val_loader, f"seed{seed}_s3",
              cfg.stage3_epochs, cfg.lr_heads_stage3, cfg.lr_backbone_stage3,
              "all")
    metrics = evaluate_modalities(m, test_loader, ("V", "A", "T"))
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--out", type=str,
                   default="results/tables/table_10_seed_reproducibility.csv")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, val_loader, test_loader = build_loaders(cfg)
    per_seed = [_train_one_seed(s, train_loader, val_loader, test_loader)
                for s in SEEDS]

    df = pd.DataFrame(per_seed)
    METRICS = [
        ("O (Openness)",          "macc_openness"),
        ("C (Conscientiousness)", "macc_conscientiousness"),
        ("E (Extraversion)",      "macc_extraversion"),
        ("A (Agreeableness)",     "macc_agreeableness"),
        ("N (Neuroticism)",       "macc_neuroticism"),
        ("mACC",                  "macc_mean"),
        ("CCC",                   "ccc_mean"),
    ]
    n = len(SEEDS)
    rows = []
    for label, k in METRICS:
        vals = df[k].values.astype(float)
        mu, sd = vals.mean(), vals.std(ddof=1)
        half = T_4 * sd / math.sqrt(n)
        rows.append({
            "Метрика": label,
            "Среднее": round(float(mu), 3),
            "Std":     round(float(sd), 3),
            "95% CI":  f"[{mu-half:.3f}; {mu+half:.3f}]",
        })

    out = pd.DataFrame(rows).set_index("Метрика")
    print(out.to_string())
    save_csv(out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
