# Headline results

> All numbers are on the FIv2 test split.  mACC = 1 − MAE (averaged
> over the five Big Five traits); CCC = mean concordance correlation
> coefficient.

## Table 1 — Modality ablation

| Configuration             | O    | C    | E    | A    | N    | mACC | CCC  |
| ------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| V (visual only)           | .916 | .923 | .913 | .920 | .916 | .918 | .652 |
| V + T                     | .920 | .925 | .918 | .922 | .919 | .921 | .670 |
| V + A                     | .924 | .929 | .925 | .925 | .926 | .926 | .708 |
| V + A + T (full, ours)    | .928 | .932 | .927 | .928 | .928 | .929 | .721 |

![Modality ablation](../results/figures/figure_1_modality_ablation.png)

## Table 3 — Computational characteristics

| Method                          | Params (M) | FLOPs (G/clip) | GPU lat. (ms) | CPU lat. (ms) | Peak mem (MB) |
| ------------------------------- | ---------- | -------------- | ------------- | ------------- | ------------- |
| EmoFormer (V)                   | 78         | 380            | 89            | 1180          | 1850          |
| GSFN (V+A+T)                    | 134        | 540            | 142           | 2310          | 2640          |
| CAT-BE (multi)                  | 156        | 720            | 168           | 2680          | 3120          |
| **VSSD-Trait V (ours)**         | **40**     | 504            | 108           | 1960          | **1680**      |
| **VSSD-Trait V+A (ours)**       | **60**     | 540            | 148           | 2680          | 2310          |
| **VSSD-Trait V+A+T (ours)**     | **63**     | 545            | 155           | 2760          | 2440          |

![Efficiency frontier](../results/figures/figure_3_efficiency_frontier.png)

## Table 11 — Final comparison

| Method                | Mod.            | mACC          | CCC          |
| --------------------- | --------------- | ------------- | ------------ |
| EmoFormer (2024)      | F               | .916          | .634         |
| CAT-BE (2022)         | F,S,A,T,M,BE    | .926          | —            |
| GSFN (2024)           | F,A,T           | .928          | .734         |
| SSL-MEPR (2026)       | F,B,S,A,T       | .929          | .781         |
| **VSSD-Trait (ours)** | F,A,T           | **.929 ± .002** | .721 ± .005 |

VSSD-Trait reaches the corpus-best mACC at 2–2.5× fewer trainable
parameters than the closest baselines.  The full set of 11 tables and
12 figures is in [`results/`](../results/).

## See also

* [`docs/architecture.md`](architecture.md) — model, losses, schedule.
* [`docs/data.md`](data.md) — corpus and feature-cache layout.
* [`docs/experiments.md`](experiments.md) — script ↔ table ↔ figure map.
* [`notebooks/FIv2_VSSD_extended_experiments_colab.ipynb`](../notebooks/FIv2_VSSD_extended_experiments_colab.ipynb)
  — runnable notebook with the same code as the scripts.
