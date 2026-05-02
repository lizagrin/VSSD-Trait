"""PyTorch dataset and dataloader factory."""
from __future__ import annotations

import os
import random
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..config import cfg
from .caching import get_cache_path
from .perturbations import (
    Perturb,
    apply_audio_noise,
    apply_jpeg,
    apply_fps,
    apply_occlusion,
)


class FaceAugment:
    """Light face-image augmentation used during training."""

    def __init__(self, train: bool = True):
        self.train = train

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        img = x.astype(np.uint8)
        if self.train:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            if random.random() < 0.5:
                a = 1.0 + random.uniform(-0.08, 0.08)
                b = random.uniform(-8, 8)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)


class FIv2CacheDataset(Dataset):
    """Reads pre-cached .pt payloads produced by :mod:`av_traits.data.caching"""

    def __init__(
        self,
        manifest: pd.DataFrame,
        train: bool = True,
        perturb: Optional[Perturb] = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.train = train
        self.face_tf = FaceAugment(train=train)
        self.perturb = perturb or Perturb()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        cp = get_cache_path(cfg.cache_root, row["split"], row["video_id"])
        if not os.path.exists(cp):
            raise FileNotFoundError(
                f"Cache file missing: {cp}. Run scripts/precache_features.py first."
            )
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        faces, scores = obj["faces"], obj["face_scores"]
        mel, prosody, ege = obj["mel"], obj["prosody"], obj["ege_maps"]
        target = obj["target"]; text = obj.get("transcription", "")

        # ---- on-the-fly perturbations (Tables 4-7)
        if self.perturb.audio_snr_db is not None:
            mel, prosody = apply_audio_noise(mel, prosody, self.perturb.audio_snr_db)
        if self.perturb.jpeg_quality is not None:
            faces = apply_jpeg(faces, self.perturb.jpeg_quality)
        if self.perturb.fps_target is not None:
            faces, scores = apply_fps(faces, scores, self.perturb.fps_target)
        if self.perturb.occlusion is not None:
            faces = apply_occlusion(faces, self.perturb.occlusion)

        faces_t = torch.stack(
            [
                torch.stack(
                    [self.face_tf(faces[w, k]) for k in range(faces.shape[1])], dim=0
                )
                for w in range(faces.shape[0])
            ],
            dim=0,
        )

        return {
            "video_id": row["video_id"],
            "faces": faces_t,
            "face_scores": torch.tensor(scores, dtype=torch.float32),
            "mel": torch.tensor(mel, dtype=torch.float32),
            "prosody": torch.tensor(prosody, dtype=torch.float32),
            "ege_maps": torch.tensor(
                ege if ege is not None else np.zeros(88, np.float32),
                dtype=torch.float32,
            ),
            "target": torch.tensor(target, dtype=torch.float32),
            "text": text,
        }


def _pad_mel_batch(mels):
    T = max(x.shape[-1] for x in mels)
    out = []
    for x in mels:
        if x.shape[-1] < T:
            x = torch.cat([x, torch.zeros(x.shape[0], T - x.shape[-1])], dim=-1)
        out.append(x)
    return torch.stack(out)


def collate_fn(batch):
    return {
        "video_id":    [b["video_id"] for b in batch],
        "text":        [b["text"]     for b in batch],
        "faces":       torch.stack([b["faces"]       for b in batch]),
        "face_scores": torch.stack([b["face_scores"] for b in batch]),
        "mel":         _pad_mel_batch([b["mel"]      for b in batch]),
        "prosody":     torch.stack([b["prosody"]     for b in batch]),
        "ege_maps":    torch.stack([b["ege_maps"]    for b in batch]),
        "target":      torch.stack([b["target"]      for b in batch]),
    }


def make_loader(manifest: pd.DataFrame, train: bool = True,
                perturb: Optional[Perturb] = None, batch_size: Optional[int] = None
                ) -> DataLoader:
    ds = FIv2CacheDataset(manifest, train=train, perturb=perturb)
    return DataLoader(
        ds,
        batch_size=batch_size or (cfg.batch_size if train else cfg.eval_batch_size),
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
