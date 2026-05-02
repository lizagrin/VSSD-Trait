"""Project-wide configuration object"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import yaml


@dataclass
class CFG:
    # paths
    project_root: str = "/content/fiv2_vssd_project"
    dataset_source: str = "local"
    local_dataset_root: str = "/content/drive/MyDrive/FIv2_persistent_cache/DATA"
    cache_root: str = "/content/fiv2_cache_local"
    checkpoints_dir: str = "/content/fiv2_checkpoints"
    logs_dir: str = "/content/fiv2_logs"
    vssd_repo_dir: str = "/content/VSSD"

    # data
    train_windows: int = 8
    eval_windows: int = 14
    train_frames_per_window: int = 4
    eval_frames_per_window: int = 8
    local_window_span_sec: float = 2.0
    face_size: int = 224
    sample_rate: int = 16000

    # audio
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 1024
    use_ege_maps: bool = True

    # text
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384

    # model
    trait_names: Tuple[str, ...] = (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    )
    num_traits: int = 5
    num_bins: int = 3

    vssd_variant: str = "small"
    vssd_ckpt_repo: str = "YuhengSSS/VSSD_ICCV_weights"
    vssd_small_ckpt: str = "vssd_small_mesa.pth"
    vssd_tiny_ckpt: str = "vssd_tiny_mesa.pth"

    hidden_dim: int = 512
    audio_deep_dim: int = 256
    audio_hc_dim: int = 128
    dropout: float = 0.15
    modality_dropout_p: float = 0.10

    # experiment switches
    active_modalities: Tuple[str, ...] = ("V", "A", "T")
    backbone_kind: str = "vssd"
    pooling_mode: str = "attention"
    fusion_mode: str = "weighted"
    use_windowing: bool = True
    use_aux_heads: bool = True
    use_ccc_loss: bool = True

    # optimization
    batch_size: int = 2
    eval_batch_size: int = 2
    num_workers: int = 4
    weight_decay: float = 1e-4
    amp: bool = True
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0

    stage1_epochs: int = 2
    stage2_epochs: int = 4
    stage3_epochs: int = 6
    lr_heads_stage1: float = 3e-4
    lr_heads_stage2: float = 1.5e-4
    lr_heads_stage3: float = 8e-5
    lr_backbone_stage2: float = 8e-6
    lr_backbone_stage3: float = 3e-6

    # losses
    lambda_ccc: float = 0.5
    lambda_bin: float = 0.2
    lambda_pair: float = 0.1
    lambda_aux_modal: float = 0.15
    huber_beta: float = 0.05

    # misc
    seed: int = 42
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "CFG":
        """Load a configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # tuples in YAML come back as lists — coerce known fields back
        for k in ("trait_names", "active_modalities"):
            if k in data and isinstance(data[k], list):
                data[k] = tuple(data[k])
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True, sort_keys=False)

    def update(self, **kwargs) -> "CFG":
        """Return a new CFG with the given overrides applied (immutable style)."""
        data = asdict(self)
        data.update(kwargs)
        return CFG(**data)


cfg = CFG()
