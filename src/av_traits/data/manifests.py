"""Build manifest DataFrames from the local FIv2 dump"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import cfg

SPLIT_MAP = {
    "train":      {"dir": "TRAIN",      "ann": "annotation_training.pkl",   "tr": "transcription_training.pkl"},
    "validation": {"dir": "VALIDATION", "ann": "annotation_validation.pkl", "tr": "transcription_validation.pkl"},
    "test":       {"dir": "TEST",       "ann": "annotation_test.pkl",       "tr": "transcription_test.pkl"},
}


def load_pickle_compat(path: str):
    """Load a pickle file, falling back to ``latin1`` for Py2-produced files."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def _get_by_video_name(mapping: Optional[dict], video_name: str):
    if not isinstance(mapping, dict):
        return None
    stem = Path(video_name).stem
    for key in (stem, video_name):
        if key in mapping:
            return mapping[key]
    return None


def get_split_paths(root: str, split: str):
    spec = SPLIT_MAP[split]
    split_dir = os.path.join(root, spec["dir"])
    ann_path  = os.path.join(split_dir, "Annotation", spec["ann"])
    tr_path   = os.path.join(split_dir, "Annotation", spec["tr"])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not os.path.exists(tr_path):
        tr_path = None
    return split_dir, ann_path, tr_path


def build_manifest_from_local(root: str) -> Dict[str, pd.DataFrame]:
    """Walk every split under ``root`` and return one DataFrame per split."""
    manifests: Dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test"]:
        split_dir, ann_path, tr_path = get_split_paths(root, split)
        ann = load_pickle_compat(ann_path)
        tr  = load_pickle_compat(tr_path) if tr_path else None

        files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            files.extend(glob.glob(os.path.join(split_dir, ext)))

        rows, skipped = [], 0
        for vp in sorted(files):
            vn = os.path.basename(vp); vid = Path(vp).stem
            row = {
                "split": split,
                "video_id": vid,
                "video_path": vp,
                "transcription": (_get_by_video_name(tr, vn) or "") if tr else "",
            }
            ok = True
            for trait in cfg.trait_names:
                v = _get_by_video_name(ann.get(trait, {}), vn) if isinstance(ann, dict) else None
                if v is None:
                    ok = False
                    break
                row[trait] = float(v)
            if ok:
                rows.append(row)
            else:
                skipped += 1

        manifests[split] = pd.DataFrame(rows)
        print(f"{split}: {manifests[split].shape}, skipped={skipped}")
    return manifests
