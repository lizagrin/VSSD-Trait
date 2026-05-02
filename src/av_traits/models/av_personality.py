"""End-to-end multimodal Big-Five model."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import cfg
from .audio import AudioModule
from .fusion import BinHead, TraitWiseLateFusion
from .text import TextModule, encode_texts_with_sbert
from .visual import VisualTemporalModule


class AVTPersonalityModel(nn.Module):
    """Visual + Audio + Text model with switchable modalities"""

    def __init__(self, backbone: nn.Module, visual_dim: int,
                 prosody_dim: int, ege_dim: int):
        super().__init__()
        self.modalities = list(cfg.active_modalities)
        if "V" in self.modalities:
            self.visual = VisualTemporalModule(backbone, visual_dim,
                                               cfg.hidden_dim, cfg.dropout)
        if "A" in self.modalities:
            self.audio = AudioModule(prosody_dim, ege_dim,
                                     cfg.hidden_dim, cfg.dropout)
        if "T" in self.modalities:
            self.text = TextModule(cfg.text_dim, cfg.hidden_dim, cfg.dropout)
        self.fusion = TraitWiseLateFusion(cfg.hidden_dim, cfg.num_traits,
                                          self.modalities)
        self.bin_head = BinHead(cfg.hidden_dim * len(self.modalities),
                                cfg.num_traits, cfg.num_bins)

    def forward(self, batch: Dict) -> Dict:
        feats: Dict[str, torch.Tensor] = {}
        info: Dict[str, Dict] = {}
        if "V" in self.modalities:
            v, vinf = self.visual(batch["faces"], batch["face_scores"])
            feats["V"] = v
            info["visual"] = vinf
        if "A" in self.modalities:
            feats["A"] = self.audio(batch["mel"], batch["prosody"], batch["ege_maps"])
        if "T" in self.modalities:
            with torch.no_grad():
                temb = encode_texts_with_sbert(batch["text"])
            feats["T"] = self.text(temb)

        pred, per_mod, alpha = self.fusion(feats)
        bin_logits = self.bin_head(torch.cat([feats[m] for m in self.modalities], dim=-1))
        return {
            "pred": pred,
            "per_mod": per_mod,
            "alpha": alpha,
            "bin_logits": bin_logits,
            "feats": feats,
            **info,
        }
