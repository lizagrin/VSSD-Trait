"""Reproduce architectural ablation"""
from __future__ import annotations

import argparse
import gc

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
from av_traits.utils.io import load_checkpoint, save_csv

VARIANTS = [
    ("VSSD → ResNet-50",                              {"backbone_kind": "resnet50"}),
    ("VSSD → ViT-B/16",                               {"backbone_kind": "vit_b16"}),
    ("Без оконного представления (clip-level)",       {"use_windowing": False}),
    ("Mean pooling вместо attention pooling",         {"pooling_mode": "mean"}),
    ("Concat вместо взвешенного слияния",             {"fusion_mode": "concat"}),
    ("Без вспомогательных классификационных голов",   {"use_aux_heads": False}),
    ("Без CCC-loss (только Huber)",                   {"use_ccc_loss": False}),
]


def _train_variant(name, overrides, train_loader, val_loader, test_loader):
    saved = {k: getattr(cfg, k) for k in overrides}
    for k, v in overrides.items():
        setattr(cfg, k, v)
    try:
        m = build_model_for_cfg(cfg, train_loader)
        for stage, ep, lh, lb, mode in [
            ("s1", cfg.stage1_epochs, cfg.lr_heads_stage1, 0.0, "frozen"),
            ("s2", cfg.stage2_epochs, cfg.lr_heads_stage2, cfg.lr_backbone_stage2, "last_stage"),
            ("s3", cfg.stage3_epochs, cfg.lr_heads_stage3, cfg.lr_backbone_stage3, "all"),
        ]:
            fit_stage(m, train_loader, val_loader,
                      f"{name}_{stage}", ep, lh, lb, mode)
        metrics = evaluate_modalities(m, test_loader, cfg.active_modalities)
        return metrics
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Reference V+A+T checkpoint (used for the baseline row).")
    p.add_argument("--out", type=str,
                   default="results/tables/table_2_architectural_ablation.csv")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, val_loader, test_loader = build_loaders(cfg)

    base_model = build_model_for_cfg(cfg, train_loader)
    load_checkpoint(base_model, args.checkpoint)
    full = evaluate_modalities(base_model, test_loader, ("V", "A", "T"))
    base_macc = full["macc_mean"]

    rows = []
    for name, ov in VARIANTS:
        print(f"\n--- Variant: {name} ({ov}) ---")
        m = _train_variant(name.replace(" ", "_").replace("→", "to")[:32],
                           ov, train_loader, val_loader, test_loader)
        rows.append({
            "Конфигурация": name,
            "mACC":  round(m["macc_mean"], 3),
            "CCC":   round(m["ccc_mean"], 3),
            "ΔmACC": round(m["macc_mean"] - base_macc, 3),
        })
    rows.append({
        "Конфигурация": "Полный метод (ours)",
        "mACC":  round(base_macc, 3),
        "CCC":   round(full["ccc_mean"], 3),
        "ΔmACC": 0.000,
    })
    df = pd.DataFrame(rows).set_index("Конфигурация")
    print(df.to_string())
    save_csv(df, args.out)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
