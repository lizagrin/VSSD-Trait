"""Train the full VSSD-Trait V+A+T model with the staged unfreezing schedule"""
from __future__ import annotations

import argparse
import os

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.training import fit_stage
from av_traits.utils.seeds import set_all_seeds


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg_ = load_runtime_cfg(args)
    set_all_seeds(cfg_.seed)

    train_loader, val_loader, _ = build_loaders(cfg_)
    model = build_model_for_cfg(cfg_, train_loader)

    fit_stage(model, train_loader, val_loader, "stage1_frozen",
              cfg_.stage1_epochs, cfg_.lr_heads_stage1, 0.0, "frozen")
    fit_stage(model, train_loader, val_loader, "stage2_last_stage",
              cfg_.stage2_epochs, cfg_.lr_heads_stage2, cfg_.lr_backbone_stage2,
              "last_stage")
    fit_stage(model, train_loader, val_loader, "stage3_all",
              cfg_.stage3_epochs, cfg_.lr_heads_stage3, cfg_.lr_backbone_stage3,
              "all")

    print("\nDone. Best checkpoint:",
          os.path.join(cfg.checkpoints_dir, "best_stage3_all.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
