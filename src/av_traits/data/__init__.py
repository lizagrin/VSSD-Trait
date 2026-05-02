"""Data utilities"""
from .perturbations import (  # noqa: F401  (re-export)
    Perturb,
    apply_audio_noise,
    apply_jpeg,
    apply_fps,
    apply_occlusion,
)


def __getattr__(name):
    if name in ("FIv2CacheDataset", "FaceAugment", "collate_fn", "make_loader"):
        from . import dataset
        return getattr(dataset, name)
    if name in ("build_manifest_from_local", "load_pickle_compat"):
        from . import manifests
        return getattr(manifests, name)
    raise AttributeError(name)


__all__ = [
    "Perturb",
    "apply_audio_noise",
    "apply_jpeg",
    "apply_fps",
    "apply_occlusion",
    "FIv2CacheDataset",
    "FaceAugment",
    "collate_fn",
    "make_loader",
    "build_manifest_from_local",
    "load_pickle_compat",
]
