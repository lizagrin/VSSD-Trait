"""Input-space perturbations used for the robustness experiments"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Perturb:
    audio_snr_db: Optional[float] = None
    jpeg_quality: Optional[int] = None
    fps_target: Optional[int] = None
    occlusion: Optional[str] = None


def apply_audio_noise(mel: np.ndarray, prosody: np.ndarray, snr_db: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to the log-mel domain at the requested SNR."""
    sig_power = float((mel ** 2).mean())
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*mel.shape).astype(mel.dtype) * np.sqrt(noise_power)
    mel_p = mel + noise
    pros = prosody.copy()
    pros[0] = pros[0] * (1 + 0.05 * np.random.randn())
    return mel_p, pros


def apply_jpeg(faces: np.ndarray, q: int) -> np.ndarray:
    """JPEG-encode → decode every face in ``[W, K, H, W, 3]``."""
    out = np.empty_like(faces)
    for w in range(faces.shape[0]):
        for k in range(faces.shape[1]):
            ok, enc = cv2.imencode(".jpg", faces[w, k],
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
            out[w, k] = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else faces[w, k]
    return out


def apply_fps(faces: np.ndarray, scores: np.ndarray, fps_target: int,
              fps_orig: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate frames inside each window to mimic a lower frame-rate."""
    step = max(1, fps_orig // max(1, fps_target))
    if step == 1:
        return faces, scores
    return faces[:, ::step], scores[:, ::step]


def apply_occlusion(faces: np.ndarray, kind: str) -> np.ndarray:
    out = faces.copy()
    H, W = faces.shape[2], faces.shape[3]
    for w in range(out.shape[0]):
        for k in range(out.shape[1]):
            if kind == "mask":
                out[w, k, H // 2:, :, :] = 128
            elif kind == "glasses":
                out[w, k, H // 4:H // 2, :, :] = 32
            elif kind == "center":
                y1, y2 = H // 4, 3 * H // 4
                x1, x2 = W // 4, 3 * W // 4
                out[w, k, y1:y2, x1:x2, :] = 128
            else:
                raise ValueError(f"Unknown occlusion kind: {kind!r}")
    return out
