# Experiments

This document maps each table in
[`Эксперименты_расширенные.docx`](../docs/Эксперименты_расширенные.docx)
(the dissertation extended experimental section) to a runnable script
and to the artifacts under [`results/`](../results/).

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
| Train full V+A+T           | ~6 h               |
| Modality ablation (T1)     | ~10 min (no train) |
| Architectural ablation (T2)| ~36 h (7 retrains) |
| Compute benchmarks (T3)    | ~5 min             |
| Robustness (T4–T7)         | ~25 min            |
| Fusion weights (T8)        | ~5 min             |
| Window attention (T9)      | ~5 min             |
| 5-seed reproducibility (T10)| ~30 h (5 retrains) |
| Final comparison (T11)     | < 1 s              |
