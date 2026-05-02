"""Reproduce final comparison with the literature"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from av_traits.utils.io import load_csv, save_csv

LIT_BASELINES = pd.DataFrame([
    ("EmoFormer (2024)",   "F",          .914, .920, .918, .914, .912, ".916", ".634"),
    ("CAT-BE (2022)",      "F,S,A,T,M,BE",.929, .926, .927, .929, .921, ".926", "—"),
    ("GSFN (2024)",        "F,A,T",      .925, .930, .932, .926, .928, ".928", ".734"),
    ("SSL-MEPR (2026)",    "F,B,S,A,T",  np.nan, np.nan, np.nan, np.nan, np.nan,
                                          ".929", ".781"),
], columns=["Метод", "Mod.", "O", "C", "E", "A", "N", "mACC", "CCC"]).set_index("Метод")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-table",
                   default="results/tables/table_10_seed_reproducibility.csv")
    p.add_argument("--out",
                   default="results/tables/table_11_final_comparison.csv")
    args = p.parse_args()

    t10 = load_csv(args.seed_table)
    fmt = lambda mu, sd: f"{mu:.3f} ± {sd:.3f}"
    macc_cell = fmt(t10.loc["mACC", "Среднее"], t10.loc["mACC", "Std"])
    ccc_cell  = fmt(t10.loc["CCC",  "Среднее"], t10.loc["CCC",  "Std"])
    ours = pd.DataFrame([{
        "Метод": "VSSD-Trait (ours)",
        "Mod.":  "F,A,T",
        "O": round(t10.loc["O (Openness)",          "Среднее"], 3),
        "C": round(t10.loc["C (Conscientiousness)", "Среднее"], 3),
        "E": round(t10.loc["E (Extraversion)",      "Среднее"], 3),
        "A": round(t10.loc["A (Agreeableness)",     "Среднее"], 3),
        "N": round(t10.loc["N (Neuroticism)",       "Среднее"], 3),
        "mACC": macc_cell, "CCC": ccc_cell,
    }]).set_index("Метод")

    df = pd.concat([LIT_BASELINES, ours])
    print(df.to_string())
    save_csv(df, args.out)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
