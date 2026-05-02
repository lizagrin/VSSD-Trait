"""Sanity checks on the input-space perturbations used in Tables 4–7"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from av_traits.data.perturbations import (
    Perturb,
    apply_audio_noise,
    apply_fps,
    apply_jpeg,
    apply_occlusion,
)


def _fake_faces(W=2, K=4, H=64, Wi=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(W, K, H, Wi, 3), dtype=np.uint8)


def test_audio_noise_lowers_snr():
    rng = np.random.default_rng(0)
    mel = rng.standard_normal((128, 200)).astype(np.float32) * 5.0
    pros = rng.standard_normal((38,)).astype(np.float32)

    sig_p = float((mel ** 2).mean())
    mel2, _ = apply_audio_noise(mel, pros, snr_db=10)
    diff = mel2 - mel
    noise_p = float((diff ** 2).mean())
    # 10 dB SNR ⇒ noise power ≈ signal_power / 10
    assert noise_p == pytest.approx(sig_p / 10.0, rel=0.4)


def test_jpeg_does_not_change_shape_or_dtype():
    f = _fake_faces()
    out = apply_jpeg(f, q=30)
    assert out.shape == f.shape
    assert out.dtype == f.dtype
    # JPEG with q=30 should perturb most pixels
    assert (out != f).mean() > 0.1


def test_fps_decimation():
    f = _fake_faces(W=3, K=10)
    s = np.ones((3, 10), dtype=np.float32)
    out_f, out_s = apply_fps(f, s, fps_target=5, fps_orig=25)
    # 25 // 5 = 5 → keep every 5th frame
    assert out_f.shape == (3, 2, 64, 64, 3)
    assert out_s.shape == (3, 2)


@pytest.mark.parametrize("kind", ["mask", "glasses", "center"])
def test_occlusion_changes_pixels(kind):
    f = _fake_faces()
    out = apply_occlusion(f, kind)
    assert out.shape == f.shape
    assert (out != f).any()


def test_perturb_dataclass_defaults():
    p = Perturb()
    assert p.audio_snr_db is None
    assert p.jpeg_quality is None
    assert p.fps_target is None
    assert p.occlusion is None
