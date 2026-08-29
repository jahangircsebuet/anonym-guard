#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/guardbreach-artifact"
mkdir -p "${PROJECT_ROOT}/results/k_aware_analysis"

python "${PROJECT_ROOT}/src/k_aware_result_analysis.py" \
  --inputs "${PROJECT_ROOT}/data/classified/*/*.jsonl" \
  --outdir "${PROJECT_ROOT}/results/k_aware_analysis" \
  --expected_k 1 2 3 \
  --qwen_controversial_as_unsafe \
  --min_n_per_language_model 3 \
  --top_k_models_per_language 10