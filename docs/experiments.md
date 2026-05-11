# Experiments

This document maps each table in the dissertation's extended
experimental section (Chapter 3) to a runnable script and to the
artifacts under [`results/`](../results/).

| #  | Table                                       | Script                               | CSV                                                | Figure                                          |
| -- | ------------------------------------------- | ------------------------------------ | -------------------------------------------------- | ----------------------------------------------- |
| 1  | Modality ablation (V / V+T / V+A / V+A+T)   | `run_modality_ablation.py`           | [`tables/table_1_modality_ablation.csv`][t1]       | [`figures/figure_1_modality_ablation.png`][f1]  |
| 2  | Architectural ablation                      | `run_architectural_ablation.py`      | [`tables/table_2_architectural_ablation.csv`][t2]  | [`figures/figure_2_architectural_ablation.png`][f2] |
| 3  | Compute characteristics                     | `run_compute_benchmarks.py`          | [`tables/table_3_compute_characteristics.csv`][t3] | [`figures/figure_3_efficiency_frontier.png`][f3]|
| 4  | Audio noise (SNR ∈ {∞, 20, 10, 5} dB)      | `run_robustness.py`                  | [`tables/table_4_audio_noise.csv`][t4]             | [`figures/figure_4_audio_noise.png`][f4]        |
| 5  | JPEG compression (Q ∈ {90, 70, 50, 30})    | `run_robustness.py`                  | [`tables/table_5_jpeg_compression.csv`][t5]        | [`figures/figure_5_jpeg_compression.png`][f5]   |
| 6  | FPS decimation (FPS ∈ {15, 10, 5})         | `run_robustness.py`                  | [`tables/table_6_fps_decimation.csv`][t6]          | [`figures/figure_6_fps_decimation.png`][f6]     |
| 7  | Face occlusions (mask / glasses / center)   | `run_robustness.py`                  | [`tables/table_7_face_occlusion.csv`][t7]          | [`figures/figure_7_face_occlusion.png`][f7]     |
| 8  | Per-trait fusion weights                    | `run_fusion_analysis.py`             | [`tables/table_8_fusion_weights.csv`][t8]          | [`figures/figure_8_fusion_weights_heatmap.png`][f8] |
| 9  | Window-attention saliency                   | `run_window_analysis.py`             | [`tables/table_9_window_attention.csv`][t9]        | [`figures/figure_9_window_attention.png`][f9]   |
| 10 | 5-seed reproducibility                      | `run_seed_reproducibility.py`        | [`tables/table_10_seed_reproducibility.csv`][t10]  | [`figures/figure_10_seed_reproducibility.png`][f10] |
| 11 | Final comparison with literature            | `build_final_table.py`               | [`tables/table_11_final_comparison.csv`][t11]      | [`figures/figure_11_final_comparison.png`][f11] |

[t1]:  ../results/tables/table_1_modality_ablation.csv
[t2]:  ../results/tables/table_2_architectural_ablation.csv
[t3]:  ../results/tables/table_3_compute_characteristics.csv
[t4]:  ../results/tables/table_4_audio_noise.csv
[t5]:  ../results/tables/table_5_jpeg_compression.csv
[t6]:  ../results/tables/table_6_fps_decimation.csv
[t7]:  ../results/tables/table_7_face_occlusion.csv
[t8]:  ../results/tables/table_8_fusion_weights.csv
[t9]:  ../results/tables/table_9_window_attention.csv
[t10]: ../results/tables/table_10_seed_reproducibility.csv
[t11]: ../results/tables/table_11_final_comparison.csv
[f1]:  ../results/figures/figure_1_modality_ablation.png
[f2]:  ../results/figures/figure_2_architectural_ablation.png
[f3]:  ../results/figures/figure_3_efficiency_frontier.png
[f4]:  ../results/figures/figure_4_audio_noise.png
[f5]:  ../results/figures/figure_5_jpeg_compression.png
[f6]:  ../results/figures/figure_6_fps_decimation.png
[f7]:  ../results/figures/figure_7_face_occlusion.png
[f8]:  ../results/figures/figure_8_fusion_weights_heatmap.png
[f9]:  ../results/figures/figure_9_window_attention.png
[f10]: ../results/figures/figure_10_seed_reproducibility.png
[f11]: ../results/figures/figure_11_final_comparison.png

## Protocol

All experiments use the standard FIv2 protocol:

* training on the 6 000-clip train split only;
* hyper-parameter / checkpoint selection on the 2 000-clip val split
  by `ccc + 0.25 · macc`;
* a **single** measurement on the 2 000-clip test split with no further
  tuning;
* identical preprocessing across compared methods.

Three metrics are reported per experiment — **MAE** (per trait),
**mACC = 1 − mean MAE** (summary, primary), and **CCC** (concordance
correlation coefficient, captures correlation + bias + variance match).

## Experiment groups

### Modality contribution 

Four configurations (V, V+T, V+A, V+A+T) on the same encoder weights,
re-calibrating the fusion gate on the val split for each subset.
This isolates each modality's marginal contribution to every trait.

<p align="center">
  <img src="../results/figures/figure_1_modality_ablation.png" alt="Modality ablation" width="640">
</p>

### Architectural ablation 

Single-component swaps relative to the full V+A+T method to quantify
the cost of each design choice (VSSD → ResNet-50 / ViT-B/16, clip-level
instead of windowed visual stream, mean pooling instead of attention
pooling, plain concat instead of gated fusion, removing aux heads,
removing the CCC loss term).

<p align="center">
  <img src="../results/figures/figure_2_architectural_ablation.png" alt="Architectural ablation" width="640">
</p>

### Compute characteristics 

Trainable parameters, FLOPs per 15 s clip, GPU latency (RTX 3090, fp16)
and peak memory, compared to EmoFormer, GSFN and CAT-BE.

<p align="center">
  <img src="../results/figures/figure_3_efficiency_frontier.png" alt="Efficiency frontier" width="640">
</p>

### Robustness to input degradations 

Four zero-shot stress tests — the **clean-trained** checkpoint is
reused for all of them:

| Experiment | Degradation                                      |
| ---------- | ------------------------------------------------ |
| Table 4    | Additive Gaussian noise on audio, SNR ∈ {∞, 20, 10, 5} dB |
| Table 5    | JPEG compression on every frame, Q ∈ {90, 70, 50, 30} |
| Table 6    | FPS decimation, target FPS ∈ {25, 15, 10, 5}     |
| Table 7    | Mask / glasses / 50 % central face occlusion     |

<p align="center">
  <img src="../results/figures/figure_4_audio_noise.png" alt="Audio noise robustness" width="500">
  <img src="../results/figures/figure_5_jpeg_compression.png" alt="JPEG compression robustness" width="500">
</p>

<p align="center">
  <img src="../results/figures/figure_6_fps_decimation.png" alt="FPS decimation" width="500">
  <img src="../results/figures/figure_7_face_occlusion.png" alt="Face occlusion" width="500">
</p>

### Interpretability analyses

Modality gate weights and temporal attention weights averaged over
the test set across 5 seeds.

<p align="center">
  <img src="../results/figures/figure_8_fusion_weights_heatmap.png" alt="Fusion weights heatmap" width="480">
  <img src="../results/figures/figure_9_window_attention.png" alt="Window attention" width="560">
</p>

### Statistical reproducibility 

5 retrains with seeds `{42, 123, 456, 789, 2024}`; mean ± std ± 95 %
CI (Student's t, df = 4).

<p align="center">
  <img src="../results/figures/figure_10_seed_reproducibility.png" alt="5-seed reproducibility" width="600">
</p>

### Final comparison 

Side-by-side numbers against EmoFormer, CAT-BE, GSFN and SSL-MEPR on
the same FIv2 test split.

<p align="center">
  <img src="../results/figures/figure_11_final_comparison.png" alt="Final comparison" width="640">
</p>

## Reproducibility quick-start

The full pipeline below assumes you have downloaded First Impressions V2
(see [`docs/data.md`](data.md)), set the paths via environment
variables, and pre-cached features at least once.

```bash
export DATA_ROOT=/path/to/FIv2
export CACHE_ROOT=/path/to/cache
export CKPT_DIR=./checkpoints
export LOGS_DIR=./logs

# everything end-to-end (≥ 2 GPU-days on RTX 3090)
bash scripts/run_all_experiments.sh
```

To run individual experiments:

```bash
# train the reference model
python scripts/train_full.py --config configs/default.yaml \
    --data-root $DATA_ROOT --cache-root $CACHE_ROOT \
    --checkpoints-dir $CKPT_DIR --logs-dir $LOGS_DIR

# evaluate one ablation, e.g. mean-pool variant of Table 2
python scripts/eval_full.py --config configs/ablation_mean_pool.yaml \
    --checkpoint $CKPT_DIR/best_stage3_all.pt --modalities V,A,T
```

## What costs what

| Experiment                 | Cost (RTX 3090)    |
| -------------------------- | ------------------ |
| Pre-cache FIv2             | ~2.5 h (one-shot)  |
| Train full V+A+T           | ~18 h              |
| Modality ablation (T1)     | ~10 min (no train) |
| Architectural ablation (T2)| ~36 h (7 retrains) |
| Compute benchmarks (T3)    | ~5 min             |
| Robustness (T4–T7)         | ~25 min            |
| Fusion weights (T8)        | ~5 min             |
| Window attention (T9)      | ~5 min             |
| 5-seed reproducibility (T10)| ~90 h (5 retrains)|
| Final comparison (T11)     | < 1 s              |
