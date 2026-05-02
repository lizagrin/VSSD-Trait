"""Helpers reused by every CLI script in this folder."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch

# Make the in-tree `src/` importable when running these scripts directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from av_traits.config import CFG, cfg  # noqa: E402
from av_traits.data import build_manifest_from_local, make_loader  # noqa: E402
from av_traits.models import (  # noqa: E402
    AVTPersonalityModel,
    build_visual_backbone,
)


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config file overriding the default CFG.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override cfg.local_dataset_root.")
    p.add_argument("--cache-root", type=str, default=None,
                   help="Override cfg.cache_root.")
    p.add_argument("--checkpoints-dir", type=str, default=None)
    p.add_argument("--logs-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def load_runtime_cfg(args) -> CFG:
    """Apply CLI overrides on top of an optional YAML config."""
    base = CFG.from_yaml(args.config) if args.config else cfg
    overrides = {}
    if args.data_root:        overrides["local_dataset_root"] = args.data_root
    if args.cache_root:       overrides["cache_root"] = args.cache_root
    if args.checkpoints_dir:  overrides["checkpoints_dir"] = args.checkpoints_dir
    if args.logs_dir:         overrides["logs_dir"] = args.logs_dir
    if args.seed is not None: overrides["seed"] = args.seed
    new = base.update(**overrides) if overrides else base

    # mutate the shared instance so module-level imports see the new values
    for k, v in vars(new).items():
        setattr(cfg, k, v)
    return cfg


def _sample_batch_dims(loader) -> Tuple[int, int]:
    sample = next(iter(loader))
    return sample["prosody"].shape[1], sample["ege_maps"].shape[1]


def build_loaders(cfg_):
    manifests = build_manifest_from_local(cfg_.local_dataset_root)
    return (
        make_loader(manifests["train"], train=True),
        make_loader(manifests["validation"], train=False),
        make_loader(manifests["test"], train=False),
    )


def build_model_for_cfg(cfg_, train_loader) -> torch.nn.Module:
    backbone, vdim = build_visual_backbone(cfg_.backbone_kind)
    pdim, edim = _sample_batch_dims(train_loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AVTPersonalityModel(backbone, vdim, pdim, edim).to(device)
