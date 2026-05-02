"""Evaluate a trained checkpoint on the FIv2 test split."""
from __future__ import annotations

import argparse
import json
import os

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.eval import evaluate_modalities
from av_traits.utils.io import load_checkpoint


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a *.pt checkpoint produced by train_full.py.")
    p.add_argument("--modalities", type=str, default="V,A,T",
                   help="Comma-separated subset of {V,A,T}.")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, _, test_loader = build_loaders(cfg)
    model = build_model_for_cfg(cfg, train_loader)
    load_checkpoint(model, args.checkpoint)

    modalities = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    metrics = evaluate_modalities(model, test_loader, modalities)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
