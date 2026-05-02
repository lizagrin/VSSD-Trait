"""One-shot feature pre-cache (faces + audio mel + prosody + eGeMAPS).
Usage::
    python scripts/precache_features.py --data-root /path/to/FIv2 \\
                                       --cache-root /path/to/cache
"""
from __future__ import annotations

import argparse

from _common import add_common_args, load_runtime_cfg
from av_traits.config import cfg
from av_traits.data.caching import precache_manifest
from av_traits.data.manifests import build_manifest_from_local


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args()
    load_runtime_cfg(args)

    import opensmile
    from facenet_pytorch import MTCNN
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=cfg.face_size, margin=12, keep_all=False,
                  post_process=False, device=device)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    ) if cfg.use_ege_maps else None

    manifests = build_manifest_from_local(cfg.local_dataset_root)
    print("Caching train …"); precache_manifest(manifests["train"],      mtcnn, smile, train_mode=True)
    print("Caching val   …"); precache_manifest(manifests["validation"], mtcnn, smile, train_mode=False)
    print("Caching test  …"); precache_manifest(manifests["test"],       mtcnn, smile, train_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
