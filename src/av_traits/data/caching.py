"""Per-clip preprocessing and on-disk caching of features"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import cfg


def get_cache_paths(root: str):
    paths = {
        "mel":   os.path.join(root, "mel"),
        "video": os.path.join(root, "video"),
        "audio": os.path.join(root, "audio"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def get_cache_path(root: str, split: str, video_id: str) -> str:
    folder = os.path.join(root, "video", split)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, video_id + ".pt")

# Audio
def make_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg.n_mels,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


def make_basic_prosody(y: np.ndarray, sr: int) -> np.ndarray:
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    except Exception:
        f0 = np.zeros_like(rms)

    feats = []
    for arr in (rms, zcr, f0):
        a = np.nan_to_num(arr)
        feats += [a.mean(), a.std(), np.median(a), a.min(), a.max()]
    mfcc = np.nan_to_num(mfcc)
    feats += mfcc.mean(axis=1).tolist() + mfcc.std(axis=1).tolist()
    return np.array(feats, dtype=np.float32)


def load_audio_from_video(video_path: str, audio_cache_dir: str, sr: int = 16000):
    wav = os.path.join(audio_cache_dir, Path(video_path).stem + ".wav")
    if not os.path.exists(wav):
        os.system(f'ffmpeg -y -i "{video_path}" -ac 1 -ar {sr} "{wav}" -loglevel error')
    y, sr = librosa.load(wav, sr=sr, mono=True)
    return y, sr, wav

# Video / faces
def sample_window_centers(duration_sec: float, n_windows: int):
    if n_windows <= 1:
        return [duration_sec / 2]
    return list(np.linspace(0.75, max(0.75, duration_sec - 0.75), n_windows))


def sample_times_around(center_sec: float, span_sec: float, n_frames: int):
    return list(np.linspace(max(0.0, center_sec - span_sec / 2),
                            center_sec + span_sec / 2, n_frames))


def seconds_to_idx(sec: float, fps: float, n_frames: int) -> int:
    return max(0, min(n_frames - 1, int(round(sec * fps))))


def detect_face_or_center(frame_rgb: np.ndarray, mtcnn) -> tuple:
    try:
        out = mtcnn(frame_rgb)
        if out is not None:
            x = out.permute(1, 2, 0).cpu().numpy()
            return np.clip(x, 0, 255).astype(np.uint8), 1.0
    except Exception:
        pass
    h, w = frame_rgb.shape[:2]
    s = min(h, w)
    crop = frame_rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(crop, (cfg.face_size, cfg.face_size)), 0.0

# Top-level cache helpers
def preprocess_and_cache_video(row, mtcnn, smile, train_mode=True, overwrite=False) -> Optional[str]:
    """Compute features for one row of the manifest and store them on disk."""
    from decord import VideoReader, cpu
    paths = get_cache_paths(cfg.cache_root)
    cache_path = get_cache_path(cfg.cache_root, row["split"], row["video_id"])
    if os.path.exists(cache_path) and not overwrite:
        return cache_path

    n_w = cfg.train_windows if train_mode else cfg.eval_windows
    k_f = cfg.train_frames_per_window if train_mode else cfg.eval_frames_per_window

    try:
        vr = VideoReader(row["video_path"], ctx=cpu(0))
    except Exception:
        return None
    if len(vr) == 0:
        return None

    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 25.0
    duration = len(vr) / max(fps, 1e-6)

    faces_all, scores_all = [], []
    for c in sample_window_centers(duration, n_w):
        fs, ss = [], []
        for t in sample_times_around(c, cfg.local_window_span_sec, k_f):
            frame = vr[seconds_to_idx(t, fps, len(vr))].asnumpy()
            f, s = detect_face_or_center(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mtcnn)
            fs.append(f); ss.append(s)
        faces_all.append(np.stack(fs))
        scores_all.append(np.array(ss, dtype=np.float32))

    y, sr, wav = load_audio_from_video(row["video_path"], paths["audio"], sr=cfg.sample_rate)
    ege = None
    if smile is not None:
        try:
            ege = smile.process_file(wav).iloc[0].values.astype(np.float32)
        except Exception:
            pass

    payload = {
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "split": row["split"],
        "faces": np.stack(faces_all).astype(np.uint8),
        "face_scores": np.stack(scores_all).astype(np.float32),
        "mel": make_logmel(y, sr),
        "prosody": make_basic_prosody(y, sr),
        "ege_maps": ege,
        "target": np.array([row[t] for t in cfg.trait_names], dtype=np.float32),
        "transcription": row.get("transcription", ""),
    }
    torch.save(payload, cache_path)
    return cache_path


def precache_manifest(manifest, mtcnn, smile, train_mode=True, overwrite=False):
    ok = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        ok += int(preprocess_and_cache_video(row, mtcnn, smile, train_mode, overwrite) is not None)
    print(f"cached {ok}/{len(manifest)}")
