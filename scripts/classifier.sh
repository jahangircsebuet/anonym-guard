#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/guardbreach-artifact"
PYTHON_SCRIPT="${PROJECT_ROOT}/src/multi_guard_prompt_classifier.py"

run_classifier () {
  local model_name="$1"
  local classifier="$2"
  local k="$3"
  shift 3

  local input_file="${PROJECT_ROOT}/data/stratified/stratified_topk_k${k}_with_prompts.jsonl"
  local output_dir="${PROJECT_ROOT}/data/classified/${model_name}"
  local output_file="${output_dir}/${model_name}_dataset_k=${k}.jsonl"

  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "Running model=${model_name}, classifier=${classifier}, k=${k}"
  echo "Input : ${input_file}"
  echo "Output: ${output_file}"
  echo "============================================================"

  python "${PYTHON_SCRIPT}" \
    --input "${input_file}" \
    --output "${output_file}" \
    --classifier "${classifier}" \
    "$@"
}


# ============================================================
# GuardReasoner
# ============================================================

run_classifier guardreasoner guardreasoner 1 \
  --device auto

run_classifier guardreasoner guardreasoner 2 \
  --device auto

run_classifier guardreasoner guardreasoner 3 \
  --device auto


# ============================================================
# MD-Judge
# ============================================================

run_classifier mdjudge mdjudge 1 \
  --device auto

run_classifier mdjudge mdjudge 2 \
  --device auto

run_classifier mdjudge mdjudge 3 \
  --device auto


# ============================================================
# X-Guard
# ============================================================

run_classifier xguard xguard 1 \
  --device auto \
  --max_new_tokens 512

run_classifier xguard xguard 2 \
  --device auto \
  --max_new_tokens 512

run_classifier xguard xguard 3 \
  --device auto \
  --max_new_tokens 512


# ============================================================
# AprielGuard
# ============================================================

run_classifier aprielguard aprielguard 1 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512

run_classifier aprielguard aprielguard 2 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512

run_classifier aprielguard aprielguard 3 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512


# ============================================================
# WildGuard
# ============================================================

run_classifier wildguard wildguard 1 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512

run_classifier wildguard wildguard 2 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512

run_classifier wildguard wildguard 3 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512


# ============================================================
# Nemotron
# Note: Nemotron requires an explicit CUDA device in the classifier code.
# When using CUDA_VISIBLE_DEVICES, cuda:0 maps to the visible GPU.
# ============================================================

run_classifier nemotron nemotron 1 \
  --device cuda:0 \
  --max_new_tokens 512

run_classifier nemotron nemotron 2 \
  --device cuda:0 \
  --max_new_tokens 512

run_classifier nemotron nemotron 3 \
  --device cuda:0 \
  --max_new_tokens 512


# ============================================================
# LlamaGuard3
# ============================================================

run_classifier llamaguard3 llamaguard3 1 \
  --device auto \
  --max_new_tokens 512

run_classifier llamaguard3 llamaguard3 2 \
  --device auto \
  --max_new_tokens 512

run_classifier llamaguard3 llamaguard3 3 \
  --device auto \
  --max_new_tokens 512


# ============================================================
# LlamaGuard Permissive
# ============================================================

run_classifier llamaguard_permissive llamaguard_permissive 1 \
  --device auto \
  --max_new_tokens 512 \
  --torch_dtype bfloat16

run_classifier llamaguard_permissive llamaguard_permissive 2 \
  --device auto \
  --max_new_tokens 512 \
  --torch_dtype bfloat16

run_classifier llamaguard_permissive llamaguard_permissive 3 \
  --device auto \
  --max_new_tokens 512 \
  --torch_dtype bfloat16


# ============================================================
# GPT-OSS Safeguard
# ============================================================

run_classifier gpt_oss_safeguard gpt_oss_safeguard 1 \
  --device auto \
  --max_new_tokens 512 \
  --torch_dtype bfloat16

run_classifier gpt_oss_safeguard gpt_oss_safeguard 2 \
  --device auto \
  --max_new_tokens 512 \
  --torch_dtype bfloat16

run_classifier gpt_oss_safeguard gpt_oss_safeguard 3 \
  --device auto \
  --max_new_tokens 512 \
  --torch_dtype bfloat16


# ============================================================
# CREST
# ============================================================

run_classifier crest crest 1 \
  --device auto \
  --torch_dtype auto

run_classifier crest crest 2 \
  --device auto \
  --torch_dtype auto

run_classifier crest crest 3 \
  --device auto \
  --torch_dtype auto


# ============================================================
# ShieldGemma
# ============================================================

run_classifier shieldgemma shieldgemma 1 \
  --device auto \
  --torch_dtype bfloat16 \
  --max_input_length 4096

run_classifier shieldgemma shieldgemma 2 \
  --device auto \
  --torch_dtype bfloat16 \
  --max_input_length 4096

run_classifier shieldgemma shieldgemma 3 \
  --device auto \
  --torch_dtype bfloat16 \
  --max_input_length 4096


# ============================================================
# Qwen3Guard-Gen
# ============================================================

run_classifier qwen3guard_gen qwen3guard_gen 1 \
  --device auto \
  --torch_dtype bfloat16 \
  --max_new_tokens 512

run_classifier qwen3guard_gen qwen3guard_gen 2 \
  --device auto \
  --torch_dtype bfloat16 \
  --max_new_tokens 512

run_classifier qwen3guard_gen qwen3guard_gen 3 \
  --device auto \
  --torch_dtype bfloat16 \
  --max_new_tokens 512