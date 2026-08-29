#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/guardbreach-artifact"
mkdir -p "${PROJECT_ROOT}/results/linguaguard_router_all_k"

python "${PROJECT_ROOT}/src/linguaguard_router_pipeline_k123.py" \
  --input_csv "${PROJECT_ROOT}/results/k_aware_analysis/analysis_ready_flat.csv" \
  --outdir "${PROJECT_ROOT}/results/linguaguard_router_all_k" \
  --k_values 1 2 3 \
  --seed 42 \
  --test_size 0.30 \
  --top_r 3