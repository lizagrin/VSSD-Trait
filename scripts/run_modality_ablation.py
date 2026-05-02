"""Reproduce modality ablation (V / V+T / V+A / V+A+T)"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.eval import evaluate_modalities
from av_traits.utils.io import load_checkpoint, save_csv

CONFIGS = [
    ("V (только видео)",       ("V",)),
    ("V + T",                  ("V", "T")),
    ("V + A",                  ("V", "A")),
    ("V + A + T (full, ours)", ("V", "A", "T")),
]


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str,
                   default="results/tables/table_1_modality_ablation.csv")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, _, test_loader = build_loaders(cfg)
    model = build_model_for_cfg(cfg, train_loader)
    load_checkpoint(model, args.checkpoint)

    rows = []
    for name, mods in CONFIGS:
        m = evaluate_modalities(model, test_loader, mods)
        rows.append({
            "Конфигурация": name,
            "O":  round(m["macc_openness"], 3),
            "C":  round(m["macc_conscientiousness"], 3),
            "E":  round(m["macc_extraversion"], 3),
            "A":  round(m["macc_agreeableness"], 3),
            "N":  round(m["macc_neuroticism"], 3),
            "mACC": round(m["macc_mean"], 3),
            "CCC":  round(m["ccc_mean"], 3),
        })
    df = pd.DataFrame(rows).set_index("Конфигурация")
    print(df.to_string())
    save_csv(df, args.out)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
