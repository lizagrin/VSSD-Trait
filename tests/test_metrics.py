"""Sanity checks on the FIv2 metrics — perfect / null predictions."""
import numpy as np
import pytest

# Skip the whole module if torch isn't available (CI runs on CPU-only image).
torch = pytest.importorskip("torch")

from av_traits.training.metrics import metrics_np, t_ccc


def test_perfect_predictions():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.2, 0.9, size=(64, 5))
    m = metrics_np(y.copy(), y.copy())
    assert m["macc_mean"] == pytest.approx(1.0, abs=1e-6)
    assert m["mae_mean"] == pytest.approx(0.0, abs=1e-6)
    # CCC of identical sequences is 1
    assert m["ccc_mean"] == pytest.approx(1.0, abs=1e-3)


def test_constant_predictions_low_ccc():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.2, 0.9, size=(64, 5))
    p = np.full_like(y, fill_value=0.5)
    m = metrics_np(p, y)
    # MAE should be small but nonzero, and CCC should collapse to ~0
    assert 0 <= m["macc_mean"] < 1.0
    assert m["ccc_mean"] == pytest.approx(0.0, abs=0.05)


def test_t_ccc_perfect():
    p = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    out = t_ccc(p, p)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)
