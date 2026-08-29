#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/guardbreach-artifact"
mkdir -p "${PROJECT_ROOT}/results/dataset_stats_by_k"

python "${PROJECT_ROOT}/src/dataset_stats_by_k.py" \
  --files \
    1="${PROJECT_ROOT}/data/stratified/stratified_topk_k1_with_prompts.jsonl" \
    2="${PROJECT_ROOT}/data/stratified/stratified_topk_k2_with_prompts.jsonl" \
    3="${PROJECT_ROOT}/data/stratified/stratified_topk_k3_with_prompts.jsonl" \
  --outdir "${PROJECT_ROOT}/data_stats"