from __future__ import annotations

import argparse
import gc
import os

import pandas as pd
import torch

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.eval import (
    count_params_M,
    measure_flops_G,
    measure_latency_ms,
    peak_memory_MB,
)
from av_traits.utils.io import load_checkpoint, load_csv, save_csv


def _measure(active, ckpt_path, train_loader, test_loader):
    cfg.active_modalities = tuple(active)
    m = build_model_for_cfg(cfg, train_loader)
    if ckpt_path:
        load_checkpoint(m, ckpt_path)

    sample = next(iter(test_loader))
    batch = {k: (v[:1] if isinstance(v, torch.Tensor) else v[:1])
             for k, v in sample.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = {
        "Параметры (M)":   count_params_M(m, only_trainable=False),
        "FLOPs (G/клип)":  measure_flops_G(m, batch),
        "GPU latency (мс)": measure_latency_ms(m, batch, device, fp16=True) if device == "cuda" else float("nan"),
        "CPU latency (мс)": measure_latency_ms(m, batch, "cpu", fp16=False, n=20, warmup=2),
        "Peak mem (МБ)":    peak_memory_MB(m, batch),
    }
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return res


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional V+A+T checkpoint to load before measuring.")
    p.add_argument("--out", type=str,
                   default="results/tables/table_3_compute_characteristics.csv")
    p.add_argument("--baselines",
                   default="results/tables/table_3_compute_characteristics.csv",
                   help="Where to read the literature baseline rows from.")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, _, test_loader = build_loaders(cfg)

    rows = []
    for name, mods in [
        ("VSSD-Trait V (ours)",          ("V",)),
        ("VSSD-Trait V+A (ours)",        ("V", "A")),
        ("VSSD-Trait V+A+T (full, ours)", ("V", "A", "T")),
    ]:
        print(f"\n{name}")
        r = _measure(mods, args.checkpoint, train_loader, test_loader)
        r["Метод"] = name
        rows.append(r)

    measured = pd.DataFrame(rows).set_index("Метод")
    if os.path.exists(args.baselines):
        baseline = load_csv(args.baselines)
        keep = [r for r in baseline.index if "ours" not in r]
        df = pd.concat([baseline.loc[keep], measured])
    else:
        df = measured
    print(df.to_string())
    save_csv(df, args.out)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
