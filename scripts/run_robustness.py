"""Reproduce robustness to input degradation"""
from __future__ import annotations

import argparse

import pandas as pd

from _common import (
    add_common_args,
    build_loaders,
    build_model_for_cfg,
    load_runtime_cfg,
)
from av_traits.config import cfg
from av_traits.data import Perturb, make_loader
from av_traits.eval import evaluate_modalities
from av_traits.utils.io import load_checkpoint, save_csv


def _eval(model, manifest, perturb, mods=("V", "A", "T")):
    return evaluate_modalities(model, make_loader(manifest, train=False, perturb=perturb), mods)


def _table_4(model, manifest):
    base_full = _eval(model, manifest, None)
    base_va   = _eval(model, manifest, None, ("V", "A"))
    rows = []
    for snr_name, snr in [("Clean (∞)", None), ("20 дБ", 20),
                          ("10 дБ", 10), ("5 дБ", 5)]:
        if snr is None:
            full_macc, va_macc = base_full["macc_mean"], base_va["macc_mean"]
        else:
            full_macc = _eval(model, manifest, Perturb(audio_snr_db=snr))["macc_mean"]
            va_macc   = _eval(model, manifest, Perturb(audio_snr_db=snr), ("V", "A"))["macc_mean"]
        rows.append({
            "SNR": snr_name,
            "V+A+T mACC":         round(full_macc, 3),
            "ΔmACC":              round(full_macc - base_full["macc_mean"], 3),
            "V+A mACC (без T)":   round(va_macc,   3),
            "Δ vs V+A clean":     round(va_macc   - base_va["macc_mean"],   3),
        })
    return pd.DataFrame(rows).set_index("SNR")


def _table_5(model, manifest):
    base = _eval(model, manifest, None)["macc_mean"]
    rows = [{"Quality factor": "Без сжатия", "mACC": round(base, 3), "ΔmACC": 0.000}]
    for q_name, q in [("Q = 90", 90), ("Q = 70", 70), ("Q = 50", 50), ("Q = 30", 30)]:
        m = _eval(model, manifest, Perturb(jpeg_quality=q))["macc_mean"]
        rows.append({"Quality factor": q_name, "mACC": round(m, 3),
                     "ΔmACC": round(m - base, 3)})
    return pd.DataFrame(rows).set_index("Quality factor")


def _table_6(model, manifest):
    base = _eval(model, manifest, None)["macc_mean"]
    rows = [{"FPS": "25 (ориг.)", "mACC": round(base, 3),
             "ΔmACC": 0.000, "Кадров на клип": 375}]
    for fps_name, fps, kpc in [("15", 15, 225), ("10", 10, 150), ("5", 5, 75)]:
        m = _eval(model, manifest, Perturb(fps_target=fps))["macc_mean"]
        rows.append({"FPS": fps_name, "mACC": round(m, 3),
                     "ΔmACC": round(m - base, 3), "Кадров на клип": kpc})
    return pd.DataFrame(rows).set_index("FPS")


def _table_7(model, manifest):
    base = _eval(model, manifest, None)
    base_macc = base["macc_mean"]
    base_per = {n: base[f"macc_{n}"] for n in cfg.trait_names}

    NAMES = {
        "mask":    ("Маска (нижняя половина лица)",  "Extraversion"),
        "glasses": ("Очки (область глаз)",            "Agreeableness"),
        "center":  ("Центральная окклюзия (50%)",     "Conscientiousness"),
    }

    rows = [{"Тип окклюзии": "Без окклюзии", "mACC": round(base_macc, 3),
             "ΔmACC": 0.000, "Наиболее затронутая черта": "—"}]
    for kind, (ru, _) in NAMES.items():
        m = _eval(model, manifest, Perturb(occlusion=kind))
        deltas = {n: m[f"macc_{n}"] - base_per[n] for n in cfg.trait_names}
        worst_n, worst_d = min(deltas.items(), key=lambda x: x[1])
        worst_human = {
            "openness": "Openness", "conscientiousness": "Conscientiousness",
            "extraversion": "Extraversion", "agreeableness": "Agreeableness",
            "neuroticism": "Neuroticism",
        }[worst_n]
        rows.append({
            "Тип окклюзии": ru,
            "mACC": round(m["macc_mean"], 3),
            "ΔmACC": round(m["macc_mean"] - base_macc, 3),
            "Наиболее затронутая черта": f"{worst_human} ({worst_d:+.3f})",
        })
    return pd.DataFrame(rows).set_index("Тип окклюзии")


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="results/tables")
    args = p.parse_args()
    load_runtime_cfg(args)

    train_loader, _, _ = build_loaders(cfg)
    model = build_model_for_cfg(cfg, train_loader)
    load_checkpoint(model, args.checkpoint)

    from av_traits.data.manifests import build_manifest_from_local
    test_manifest = build_manifest_from_local(cfg.local_dataset_root)["test"]

    for name, fn in [
        ("table_4_audio_noise",      _table_4),
        ("table_5_jpeg_compression", _table_5),
        ("table_6_fps_decimation",   _table_6),
        ("table_7_face_occlusion",   _table_7),
    ]:
        print(f"\n=== {name} ===")
        df = fn(model, test_manifest)
        print(df.to_string())
        save_csv(df, f"{args.out_dir}/{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
