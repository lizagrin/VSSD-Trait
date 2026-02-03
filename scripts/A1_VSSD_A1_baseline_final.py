#%% md
# # A1. VSSD-Video baseline (mean pooling)
# 
# Эксперимент **A1** из плана:
# - окна: **2с**, шаг **1с**, **K=8** кадров на окно  
# - пулинг по кадрам в окне: **mean**  
# - агрегация по окнам: **mean + std**  
# - loss: **Huber (SmoothL1Loss)**  
# 
# Цель: получить «честный» baseline на First Impressions V2.
# 
#%%
import sys
! rm -rf ~/.local/lib/python3.9/site-packages/torch*
! rm -rf ~/.local/lib/python3.9/site-packages/nvidia*
! rm -rf ~/.local/lib/python3.9/site-packages/triton*
# ОДИН РАЗ: фикс конфликтов с бинарниками под NumPy 1.x
!{sys.executable} -m pip install --user --no-cache-dir "numpy<2" --force-reinstall
#%%
from pathlib import Path
import sys, site

print("Python:", sys.executable)

# --- 1. user-site в начало sys.path ---
user_site = site.getusersitepackages()
print("USER_SITE:", user_site)

if user_site in sys.path:
    sys.path.remove(user_site)
sys.path.insert(0, user_site)

print("sys.path[0]:", sys.path[0])
#%%
import sys

# 1. Базовые пакеты (устанавливаются в ~/.local один раз)
!{sys.executable} -m pip install --user --no-cache-dir \
    "torch==2.1.0" torchvision torchaudio \
    "timm==0.4.12" pytest chardet termcolor submitit \
    tensorboardX fvcore seaborn opencv-python tensorboard einops

# 2. mamba-ssm ставим ОТДЕЛЬНО
!{sys.executable} -m pip install --user --no-cache-dir "mamba-ssm==2.2.5"

# Чистим кеш pip, чтобы не забивать квоту
!rm -rf ~/.cache/pip

#%%
from pathlib import Path
import sys

import torch
from torch.utils import _pytree as torch_pytree

# --- Патч для transformers: добавляем register_pytree_node, если его нет ---
if not hasattr(torch_pytree, "register_pytree_node"):
    # transformers использует эту функцию при импорте, нам достаточно заглушки
    def register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None):
        return
    torch_pytree.register_pytree_node = register_pytree_node

# --- VSSD: путь к classification ---
VSSD_CLASSIFICATION_ROOT = Path("../VSSD/classification")
assert VSSD_CLASSIFICATION_ROOT.exists(), "Укажи правильный путь к VSSD/classification"

if str(VSSD_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VSSD_CLASSIFICATION_ROOT))

from config import _C, get_config
from models import build_model as vssd_build_model

# --- Triton-опы из mamba_ssm, прокидываем в mamba2 ---
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

import models.mamba2 as mamba2

mamba2.mamba_chunk_scan_combined = mamba_chunk_scan_combined
mamba2.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
mamba2.RMSNormGated = RMSNormGated
mamba2.selective_state_update = selective_state_update

print("Импорты VSSD + mamba_ssm успешно прошли")

#%%
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.cuda.amp import autocast, GradScaler   # ДОБАВЬ ЯВНО
torch.backends.cudnn.benchmark = True

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

import matplotlib.pyplot as plt


cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 2) Paths (FIv2 structure)
# =========================
DATA_ROOT = Path("../DATA")
TRAIN_DIR = DATA_ROOT / "TRAIN"
VAL_DIR   = DATA_ROOT / "VALIDATION"

assert TRAIN_DIR.exists(), f"Missing: {TRAIN_DIR}"
assert VAL_DIR.exists(), f"Missing: {VAL_DIR}"

# =========================
# 3) A1 hyperparams
# =========================
IMG_SIZE = 224

WIN_SEC  = 2.0
STEP_SEC = 1.0
K_FRAMES = 8

# Для батчинга делаем фиксированное число окон.
# В FIv2 клипы ~15с => floor((15-2)/1)+1 = 14 окон.
NUM_WINDOWS = 14
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4  # effective batch ~4
NUM_WORKERS = 8

EPOCHS = 10
LR = 3e-5
WEIGHT_DECAY = 1e-4

HUBER_BETA = 0.5  # SmoothL1Loss(beta=...)

#%%
# ==============================
# 5) Load FIv2 targets (pkl)
# ==============================
import pickle

BIG_FIVE_ORDER = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
]

def _safe_pickle_load(path: Path):
    # Для pkl, сохранённых в Python 2, может потребоваться encoding при распаковке
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            try:
                return pickle.load(f, encoding="latin1")
            except Exception:
                f.seek(0)
                return pickle.load(f, encoding="bytes")

def _load_two_pkls(annotation_dir: Path):
    pkl_files = sorted(annotation_dir.glob("*.pkl"))
    assert len(pkl_files) == 2, f"Expected 2 pkl files in {annotation_dir}, got {len(pkl_files)}"
    objs = []
    for p in pkl_files:
        objs.append(_safe_pickle_load(p))
    return objs

def load_annotation_and_transcription(phase_dir: Path):
    annot_dir = phase_dir / "Annotation"
    assert annot_dir.exists(), f"Missing: {annot_dir}"
    a, b = _load_two_pkls(annot_dir)

    # annotation: dict(trait -> dict(video->float))
    # transcription: dict(video->str)
    def looks_like_annotation(x):
        return (
            isinstance(x, dict)
            and all(k in x for k in BIG_FIVE_ORDER)
            and all(isinstance(x[k], dict) for k in BIG_FIVE_ORDER)
        )

    if looks_like_annotation(a):
        annotation, transcription = a, b
    elif looks_like_annotation(b):
        annotation, transcription = b, a
    else:
        raise RuntimeError("Could not detect annotation dict in pkl files.")

    return annotation, transcription

def build_targets_dict(annotation: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    any_trait = BIG_FIVE_ORDER[0]
    for vid_name in annotation[any_trait].keys():
        vals = [float(annotation[trait][vid_name]) for trait in BIG_FIVE_ORDER]
        targets[vid_name] = np.array(vals, dtype=np.float32)
    return targets

train_annotation, _ = load_annotation_and_transcription(TRAIN_DIR)
val_annotation, _   = load_annotation_and_transcription(VAL_DIR)

train_targets = build_targets_dict(train_annotation)
val_targets   = build_targets_dict(val_annotation)

print("Train targets:", len(train_targets))
print("Val targets  :", len(val_targets))

#%%
# ==============================
# 6) Transforms + window sampler
# ==============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_frame_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_frame_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def _make_windows_indices(
    total_frames: int,
    fps: float,
    win_sec: float,
    step_sec: float,
    k_frames: int,
    num_windows: int,
) -> List[List[int]]:
    # Возвращает ровно num_windows окон, каждое — список из k_frames индексов кадров.
    # Если реальных окон меньше — дополняем повторением последнего.
    # Если больше — равномерно подвыбираем.
    if fps <= 1e-6:
        fps = 25.0  # fallback

    win_len = max(1, int(round(win_sec * fps)))
    step_len = max(1, int(round(step_sec * fps)))

    if total_frames <= 0:
        return [[0] * k_frames for _ in range(num_windows)]

    starts = list(range(0, max(1, total_frames - win_len + 1), step_len))
    if len(starts) == 0:
        starts = [0]

    windows = []
    for s in starts:
        e = min(total_frames, s + win_len)
        if e - s <= 1:
            idxs = [min(total_frames - 1, s)] * k_frames
        else:
            idxs = np.linspace(s, e - 1, k_frames, dtype=int).tolist()
        windows.append(idxs)

    if len(windows) >= num_windows:
        pick = np.linspace(0, len(windows) - 1, num_windows, dtype=int).tolist()
        windows = [windows[i] for i in pick]
    else:
        last = windows[-1]
        while len(windows) < num_windows:
            windows.append(last)

    return windows

def load_video_windows(video_path: Path, train: bool) -> torch.Tensor:
    # Returns: frames [NUM_WINDOWS, K_FRAMES, 3, IMG_SIZE, IMG_SIZE], float32
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    windows = _make_windows_indices(
        total_frames=total,
        fps=fps,
        win_sec=WIN_SEC,
        step_sec=STEP_SEC,
        k_frames=K_FRAMES,
        num_windows=NUM_WINDOWS,
    )

    transform = train_frame_transform if train else eval_frame_transform

    out = torch.empty((NUM_WINDOWS, K_FRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)

    for wi, idxs in enumerate(windows):
        for ki, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame_bgr = cap.read()
            if not ret:
                frame_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
# frame_rgb: np.ndarray uint8 [H,W,3]
frame_rgb = np.asarray(frame_rgb)
x = torch.as_tensor(frame_rgb, dtype=torch.uint8)              # [H,W,3], uint8
x = x.permute(2, 0, 1).contiguous().to(dtype=torch.float32)    # [3,H,W], float32
x = x.div_(255.0)                                              # [0,1]
            x = transform(x)
            out[wi, ki] = x

    cap.release()
    return out

#%%
# ==============================
# 7) Dataset (windowed video)
# ==============================
class FirstImpressionsWindowDataset(Dataset):
    def __init__(self, phase_dir: Path, targets: Dict[str, np.ndarray], train: bool):
        self.phase_dir = Path(phase_dir)
        self.targets = targets
        self.train = train

        self.video_paths = sorted(self.phase_dir.glob("*.mp4"))

        # FIv2: иногда ключи targets могут быть со стемом или с .mp4
        self.video_ids: List[str] = []
        filtered = []
        for p in self.video_paths:
            stem = p.stem
            name = p.name
            key = None
            if stem in targets:
                key = stem
            elif name in targets:
                key = name
            if key is not None:
                filtered.append(p)
                self.video_ids.append(key)

        self.video_paths = filtered
        assert len(self.video_paths) == len(self.video_ids)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        vid_id = self.video_ids[idx]

        frames = load_video_windows(video_path, train=self.train)  # [W,K,3,H,W]
        target = torch.as_tensor(self.targets[vid_id], dtype=torch.float32)            # [5]

        return {"video_name": vid_id, "frames": frames, "target": target}

train_ds = FirstImpressionsWindowDataset(TRAIN_DIR, train_targets, train=True)
val_ds   = FirstImpressionsWindowDataset(VAL_DIR,   val_targets,   train=False)

print("len(train_ds) =", len(train_ds))
print("len(val_ds)   =", len(val_ds))

sample = train_ds[0]
print("frames:", sample["frames"].shape, "target:", sample["target"])

#%%
# ==============================
# 8) Model: VSSD + A1 pooling
# ==============================
class VSSDBackboneTrainable(nn.Module):
    def __init__(self, pretrained_ckpt: Optional[str] = None):
        super().__init__()
        cfg = _C.clone()
        cfg.defrost()
        cfg.MODEL.TYPE = "vmamba2"
        cfg.freeze()

        backbone = vssd_build_model(cfg)
        if backbone is None:
            raise RuntimeError("build_model(cfg) returned None. Check VSSD imports/config.")

        if pretrained_ckpt is not None:
            state = torch.load(pretrained_ckpt, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            print("Loaded ckpt. Missing:", len(missing), "Unexpected:", len(unexpected))

        self.backbone = backbone

        if hasattr(backbone, "num_features"):
            self.feat_dim = int(backbone.num_features)
        elif hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
            self.feat_dim = int(backbone.head.in_features)
        else:
            raise RuntimeError("Cannot infer feat_dim from backbone.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

class A1WindowMeanMeanStdRegressor(nn.Module):
    def __init__(self, backbone: VSSDBackboneTrainable, out_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        d = backbone.feat_dim

        self.head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, out_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, W, K, 3, H, W]
        B, Wn, K, C, H, W = frames.shape
        x = frames.reshape(B * Wn * K, C, H, W)

        # НЕ форсим fp32: пусть autocast использует fp16 (экономия памяти)
        x = x.to(dtype=torch.float16)
        feats = self.backbone.forward_features(x)  # [B*W*K, D]
D = feats.shape[-1]
        feats = feats.reshape(B, Wn, K, D)

        # A1: mean внутри окна
        win_vec = feats.mean(dim=2)  # [B, W, D]

        # A1: mean+std по окнам
        mean_w = win_vec.mean(dim=1)                       # [B, D]
        std_w  = win_vec.var(dim=1, unbiased=False).sqrt() # [B, D]
        clip_vec = torch.cat([mean_w, std_w], dim=-1)      # [B, 2D]

        return self.head(clip_vec)

backbone = VSSDBackboneTrainable(pretrained_ckpt=None).to(device)
model = A1WindowMeanMeanStdRegressor(backbone=backbone).to(device)

print("Model ready. feat_dim =", backbone.feat_dim)

#%%
# ==============================
# 9) Loss + metrics (MAE/mACC/CCC)
# ==============================
loss_fn = nn.SmoothL1Loss(beta=float(HUBER_BETA))

@torch.no_grad()
def ccc_per_trait(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: [N,5] -> CCC per trait [5]
    pred = pred.float()
    target = target.float()
    mu_x = pred.mean(dim=0)
    mu_y = target.mean(dim=0)
    vx = pred.var(dim=0, unbiased=False)
    vy = target.var(dim=0, unbiased=False)
    cov = ((pred - mu_x) * (target - mu_y)).mean(dim=0)
    ccc = (2 * cov) / (vx + vy + (mu_x - mu_y).pow(2) + eps)
    return ccc

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()

    mae_per = (pred - target).abs().mean(dim=0)  # [5]
    mae_mean = float(mae_per.mean().item())
    macc_mean = float((1.0 - mae_per).mean().item())

    ccc = ccc_per_trait(pred, target)            # [5]
    ccc_mean = float(ccc.mean().item())

    out = {
        "MAE_mean": mae_mean,
        "mACC_mean": macc_mean,
        "CCC_mean": ccc_mean,
    }
    for i, trait in enumerate(BIG_FIVE_ORDER):
        out[f"MAE_{trait}"] = float(mae_per[i].item())
        out[f"mACC_{trait}"] = float((1.0 - mae_per[i]).item())
        out[f"CCC_{trait}"] = float(ccc[i].item())
    return out

#%%
# ==============================
# 10) Train / eval loops
# ==============================
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_pred, all_targ = [], []

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True).to(dtype=torch.float16)
        target = batch["target"].to(device, non_blocking=True).to(dtype=torch.float32)

        with autocast(enabled=True):
            pred = model(frames)
            loss = loss_fn(pred, target)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        total_loss += float(loss.item()) * GRAD_ACCUM_STEPS * frames.size(0)
        all_pred.append(pred.detach().cpu())
        all_targ.append(target.detach().cpu())

        if (step % GRAD_ACCUM_STEPS) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if (len(loader) % GRAD_ACCUM_STEPS) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    epoch_loss = total_loss / len(loader.dataset)
    pred = torch.cat(all_pred, dim=0)
    targ = torch.cat(all_targ, dim=0)
    metrics = compute_metrics(pred, targ)
    return epoch_loss, metrics

def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_pred, all_targ = [], []

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True).to(dtype=torch.float16).float()
        target = batch["target"].to(device, non_blocking=True).to(dtype=torch.float32).float()

        with autocast(enabled=True):
            pred = model(frames)
            loss = loss_fn(pred, target)

        total_loss += float(loss.item()) * frames.size(0)
        all_pred.append(pred.detach().cpu())
        all_targ.append(target.detach().cpu())

    epoch_loss = total_loss / len(loader.dataset)
    pred = torch.cat(all_pred, dim=0)
    targ = torch.cat(all_targ, dim=0)
    metrics = compute_metrics(pred, targ)
    return epoch_loss, metrics

#%%
# ==============================
# 11) Run A1 training
# ==============================
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler(enabled=(device.type == "cuda"))

best_val_ccc = -1e9
os.makedirs("checkpoints_A1", exist_ok=True)

history = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_m = train_one_epoch(model, train_loader, optimizer, scaler)
    val_loss, val_m = evaluate(model, val_loader)

    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_CCC_mean": float(train_m["CCC_mean"]),
        "val_CCC_mean": float(val_m["CCC_mean"]),
        "train_mACC_mean": float(train_m["mACC_mean"]),
        "val_mACC_mean": float(val_m["mACC_mean"]),
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss {train_loss:.4f}, CCC {train_m['CCC_mean']:.4f}, mACC {train_m['mACC_mean']:.4f} | "
        f"val loss {val_loss:.4f}, CCC {val_m['CCC_mean']:.4f}, mACC {val_m['mACC_mean']:.4f}"
    )

    # сохраняем лучшую по CCC_mean (основная целевая метрика)
    if val_m["CCC_mean"] > best_val_ccc:
        best_val_ccc = float(val_m["CCC_mean"])
        ckpt_path = Path("checkpoints_A1") / "A1_best_by_CCC.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "cfg": {
                    "WIN_SEC": WIN_SEC,
                    "STEP_SEC": STEP_SEC,
                    "K_FRAMES": K_FRAMES,
                    "NUM_WINDOWS": NUM_WINDOWS,
                    "loss": "huber",
                    "huber_beta": HUBER_BETA,
                },
            },
            ckpt_path,
        )
        print("  saved best checkpoint:", ckpt_path, "| best CCC:", best_val_ccc)

print("Done.")
