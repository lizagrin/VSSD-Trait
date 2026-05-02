#!/usr/bin/env bash
# End-to-end driver: train the full model, then run every experimental
# script in order. Assumes the FIv2 cache has already been built (see
# scripts/precache_features.py).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT=/path/to/FIv2}"
CACHE_ROOT="${CACHE_ROOT:?Set CACHE_ROOT=/path/to/cache}"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
LOGS_DIR="${LOGS_DIR:-./logs}"
TABLES_DIR="${TABLES_DIR:-./results/tables}"

COMMON=(--data-root "$DATA_ROOT"
        --cache-root "$CACHE_ROOT"
        --checkpoints-dir "$CKPT_DIR"
        --logs-dir "$LOGS_DIR")

# 1. Pre-cache visual + audio features (skip if already cached)
python scripts/precache_features.py "${COMMON[@]}"

# 2. Train the full V+A+T reference model
python scripts/train_full.py "${COMMON[@]}"
BEST_CKPT="$CKPT_DIR/best_stage3_all.pt"

# 3. Modality ablation (Table 1)
python scripts/run_modality_ablation.py "${COMMON[@]}" --checkpoint "$BEST_CKPT" \
    --out "$TABLES_DIR/table_1_modality_ablation.csv"

# 4. Architectural ablation (Table 2)  -- expensive: 7 retrains
python scripts/run_architectural_ablation.py "${COMMON[@]}" --checkpoint "$BEST_CKPT" \
    --out "$TABLES_DIR/table_2_architectural_ablation.csv"

# 5. Computational characteristics (Table 3)
python scripts/run_compute_benchmarks.py "${COMMON[@]}" --checkpoint "$BEST_CKPT" \
    --out "$TABLES_DIR/table_3_compute_characteristics.csv"

# 6. Robustness experiments (Tables 4-7)
python scripts/run_robustness.py "${COMMON[@]}" --checkpoint "$BEST_CKPT" \
    --out-dir "$TABLES_DIR"

# 7. Modality fusion weights (Table 8)
python scripts/run_fusion_analysis.py "${COMMON[@]}" --checkpoint "$BEST_CKPT" \
    --out "$TABLES_DIR/table_8_fusion_weights.csv"

# 8. Window attention (Table 9)
python scripts/run_window_analysis.py "${COMMON[@]}" --checkpoint "$BEST_CKPT" \
    --out "$TABLES_DIR/table_9_window_attention.csv"

# 9. 5-seed reproducibility (Table 10) -- expensive: 5 retrains
python scripts/run_seed_reproducibility.py "${COMMON[@]}" \
    --out "$TABLES_DIR/table_10_seed_reproducibility.csv"

# 10. Final comparison table (Table 11)
python scripts/build_final_table.py \
    --seed-table "$TABLES_DIR/table_10_seed_reproducibility.csv" \
    --out "$TABLES_DIR/table_11_final_comparison.csv"

# 11. Render every figure
python scripts/make_figures.py

echo "All experiments complete. Results: $TABLES_DIR/  results/figures/"
