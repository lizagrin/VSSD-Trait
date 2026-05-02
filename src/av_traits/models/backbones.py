"""Visual backbones"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from ..config import cfg


def _vssd_args(cfg_path: str):
    return argparse.Namespace(
        cfg=cfg_path, opts=None, batch_size=None, data_path="", zip=False,
        cache_mode="part", pretrained="", resume="", accumulation_steps=None,
        use_checkpoint=False, disable_amp=False, output="/content/vssd_tmp",
        tag="default", eval=False, throughput=False, traincost=False,
        enable_persistance=False, enable_amp=True, fused_layernorm=False,
        optim=None, ddp="torch",
    )


def _list_vssd_cfgs():
    return sorted(set(
        glob.glob(os.path.join(cfg.vssd_repo_dir, "classification", "configs", "**", "*.yaml"),
                  recursive=True)
        + glob.glob(os.path.join(cfg.vssd_repo_dir, "classification", "configs", "**", "*.yml"),
                    recursive=True)
    ))


def _choose_vssd_cfg(variant: str = "small") -> str:
    files = _list_vssd_cfgs()
    if not files:
        raise FileNotFoundError("VSSD: no yaml config files found")
    scored = []
    for p in files:
        n = os.path.basename(p).lower()
        s = 0
        if variant in n: s += 10
        if "iccv2025" in n: s += 8
        if "mesa" in n: s += 5
        if variant in ("tiny", "small") and variant in n: s += 3
        if any(k in n for k in ("vmamba2", "vssd", "mamba2")): s += 2
        scored.append((s, p))
    return sorted(scored, reverse=True)[0][1]


def load_vssd_backbone(variant: str = "small") -> Tuple[nn.Module, int]:
    """Load VSSD-Small (or VSSD-Tiny) with the MESA pretrained weights."""
    cls_dir = os.path.join(cfg.vssd_repo_dir, "classification")
    if cls_dir not in sys.path:
        sys.path.insert(0, cls_dir)

    from huggingface_hub import hf_hub_download
    from config import get_config           # noqa: WPS433
    from models import build_model as vssd_build_model  # noqa: WPS433

    p = _choose_vssd_cfg(variant)
    vcfg = get_config(_vssd_args(p))
    model = vssd_build_model(vcfg)
    ckpt_name = cfg.vssd_small_ckpt if variant == "small" else cfg.vssd_tiny_ckpt
    ckpt_path = hf_hub_download(repo_id=cfg.vssd_ckpt_repo, filename=ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_ema") or state.get("model") or state.get("state_dict") or state
    miss, unx = model.load_state_dict(sd, strict=False)
    print(f"VSSD load: missing={len(miss)} unexpected={len(unx)}")

    model.head = nn.Identity()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    with torch.no_grad():
        d = torch.randn(2, 3, cfg.face_size, cfg.face_size,
                        device=next(model.parameters()).device)
        D = model.forward_features(d).shape[-1]
    print("VSSD feature dim:", D)
    return model, D


class _TimmFeatWrapper(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=True, num_classes=0,
                                     global_pool="avg")
        with torch.no_grad():
            self.feat_dim = self.net(
                torch.randn(1, 3, cfg.face_size, cfg.face_size)
            ).shape[-1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_visual_backbone(kind: str = "vssd") -> Tuple[nn.Module, int]:
    if kind == "vssd":
        return load_vssd_backbone(cfg.vssd_variant)
    if kind == "resnet50":
        m = _TimmFeatWrapper("resnet50")
        return m.to("cuda" if torch.cuda.is_available() else "cpu").eval(), m.feat_dim
    if kind == "vit_b16":
        m = _TimmFeatWrapper("vit_base_patch16_224")
        return m.to("cuda" if torch.cuda.is_available() else "cpu").eval(), m.feat_dim
    raise ValueError(f"Unknown backbone kind: {kind!r}")
