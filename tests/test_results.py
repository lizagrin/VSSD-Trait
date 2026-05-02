"""Make sure the canonical CSV / PNG artifacts are present and well-formed."""
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(ROOT, "results", "tables")
FIGS_DIR = os.path.join(ROOT, "results", "figures")
MANIFEST = os.path.join(ROOT, "results", "manifest.json")


EXPECTED_TABLES = [
    "table_1_modality_ablation",
    "table_2_architectural_ablation",
    "table_3_compute_characteristics",
    "table_4_audio_noise",
    "table_5_jpeg_compression",
    "table_6_fps_decimation",
    "table_7_face_occlusion",
    "table_8_fusion_weights",
    "table_9_window_attention",
    "table_10_seed_reproducibility",
    "table_11_final_comparison",
]


@pytest.mark.parametrize("name", EXPECTED_TABLES)
def test_table_exists_and_loads(name):
    p = os.path.join(TABLES_DIR, name + ".csv")
    assert os.path.exists(p), f"missing CSV: {p}"
    df = pd.read_csv(p, index_col=0)
    assert len(df) > 0


def test_t1_full_row_matches_paper():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_1_modality_ablation.csv"),
                     index_col=0)
    full = df.loc["V + A + T (full, ours)"]
    assert pytest.approx(0.929, abs=1e-3) == full["mACC"]
    assert pytest.approx(0.721, abs=1e-3) == full["CCC"]


def test_t10_macc_mean_and_std():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_10_seed_reproducibility.csv"),
                     index_col=0)
    row = df.loc["mACC"]
    assert pytest.approx(0.929, abs=1e-3) == row["Среднее"]
    assert pytest.approx(0.002, abs=1e-3) == row["Std"]


def test_manifest_lists_all_tables():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        m = json.load(f)
    for name in EXPECTED_TABLES:
        assert name in m["tables"]


def test_figures_directory_has_pngs():
    pngs = [n for n in os.listdir(FIGS_DIR) if n.endswith(".png")]
    assert len(pngs) >= 11, f"only {len(pngs)} figures present"
