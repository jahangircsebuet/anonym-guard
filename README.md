# GuardBreach Artifact README

This artifact accompanies the USENIX Security submission **GuardBreach: Large-Scale Multilingual LLM Safety Guardrail Evasion and Language-Aware Ensemble Defence**. It contains scripts for running multilingual dataset creation, multilingual guardrail classification, computing deployment-aware metrics, generating dataset statistics, and evaluating the LinguaGuard routing/ensemble mitigation.

---

## 1. Artifact overview

GuardBreach evaluates whether runtime LLM safety guardrails block harmful prompts when harmful intent is preserved but expressed across many non-English languages (98 languages). The paper evaluates stratified benchmark subsets corresponding to approximately 5K, 10K, and 15K prompts, denoted as `k=1`, `k=2`, and `k=3`. The main all-model comparison uses `k=2`; `k=3` is treated as a partial large-scale stress test because not all guardrail variants completed that subset.

The artifact supports the following parts of the paper:

- Multilingual unsafe prompt dataset creation along with some language metadata like language resource tier, script type etc.
- Multilingual guardrail classification over stratified `k` subsets.
- Deployment-aware result analysis using strict unsafe recall, strict evasion, coverage, unknown rate, and error rate.
- Dataset statistics and balance checks for the stratified `k` subsets.
- LinguaGuard routing and conservative top-3 ensemble evaluation.
- Generation of the tables and figures reported in the Results and Appendix.

---

## 2. Directory layout

Expected artifact layout:

```text
.
├── README.md
├── requirements.txt                  # or environment.yml
├── scripts/
│   ├── classifier.sh                 # wrapper for guardrail inference
│   ├── results.sh                    # wrapper for k-aware result analysis, from all the inference result files under classified folder
│   ├── router.sh                    # wrapper for ensemble/router (LinguaGuard) based solution for better guardrail solution 
│   ├── data.sh                      # wrapper for dataset analysis
│   ├── script.sh                    # wrapper for all the above .sh scripts
analysis
├── src/
│   ├── multi_guard_prompt_classifier.py
│   ├── k_aware_result_analysis.py
│   ├── dataset_stats_by_k.py
│   └── linguaguard_router_pipeline_k123.py
│   └── dataset-creation/ (we can provide more details upon request)
│       ├── translation_quality_based_sampling.py
│       ├── translation_quality_based_sampling.md
│       ├── translate.py
│       └── translate.md
├── data/
│   ├── stratified/
│   │   ├── stratified_topk_k1_with_prompts.jsonl
│   │   ├── stratified_topk_k2_with_prompts.jsonl
│   │   └── stratified_topk_k3_with_prompts.jsonl
│   └── Classified/
│       └── *.jsonl                   # guardrail outputs, produced by classifier.sh
├── results/
│   ├── k_aware_analysis/
│   └── linguaguard_router_all_k/
└── logs/
```

---

## 3. Environment setup

### 3.1 Hardware

The guardrail inference stage requires a CUDA-enabled GPU for most model families. The analysis, dataset-statistics, and LinguaGuard router stages can run on CPU after classification outputs are available.

### 3.2 Python environment

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some guardrail models may require gated or license-controlled access through the model provider. Reviewers should ensure that the environment has appropriate model-cache access or that precomputed classification outputs are included in `data/Classified/`.

---

## 4. Data format

Each dataset (directory: **PROJECT_ROOT/data/stratified**) row contains fields similar to:

```json
{
  "root_id": "...",
  "language": "...",
  "category": "...",
  "tier": "low|medium|high",
  "f1": "...",
  "prompt_len": "...",
  "label": "unsafe",
  "failure": "0|1",
  "severe_failure": "0|1",
  "comet": "...",
  "combined_score": "...",
  "comet_available": "0|1",
  "quality_bucket": "...",
  "selection_score": "...",
  "sampling_strategy": "...",
  "k_per_cell": "1|2|3",
  "prompt": "..."
}
```

Important terminology: the field `tier` means **language-resource tier**, not harm severity.

After guardrail classification (directory: **PROJECT_ROOT/data/classified**), each row additionally contains a nested `classifier` object:

```json
{
  "classifier": {
    "classifier_name": "qwen3guard_gen",
    "model": "Qwen/Qwen3Guard-Gen-8B",
    "label": "safe|unsafe|unknown|error",
    "raw_output": "..."
  }
}
```

---

## 5. Running the experiments

Before running any stage, set the artifact root, **the PROJECT_ROOT in each .sh file**:

```bash
PROJECT_ROOT="/path/to/guardbreach-artifact"
```

Then make execcutable all the .sh files:
```bash
chmod +x scripts/*.sh
```

### 5.1 Guardrail classification (this run will be expensive, classified data are there in classified folder)


Then create directories (after doing **cd to the PROJECT_ROOT/data directory**):

```bash
mkdir classified-new
```

Then keep active the commands for a single model you want to run the classification/inference (keep commented the other commands for the other models). Let's say you want to run the inference for model: GuardReasoner.
**keep active the GuardReasoner, keep commented the other model command like below**:
```bash
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
# # MD-Judge
# # ============================================================

# run_classifier mdjudge mdjudge 1 \
#   --device auto

# run_classifier mdjudge mdjudge 2 \
#   --device auto

# run_classifier mdjudge mdjudge 3 \
#   --device auto

# ============================================================
# # X-Guard
# # ============================================================

# run_classifier xguard xguard 1 \
#   --device auto \
#   --max_new_tokens 512

# run_classifier xguard xguard 2 \
#   --device auto \
#   --max_new_tokens 512

# run_classifier xguard xguard 3 \
#   --device auto \
#   --max_new_tokens 512
```

Then execute the script.sh file (keep one command active at a time, comment the other commands). let's say you want to run the classifier.sh then keep active the command for the classifier and keep commented other commands like below:

```bash
CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/classifier.sh" \
  >> "${PROJECT_ROOT}/logs/classifier.log" 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/results.sh" \
#   >> "${PROJECT_ROOT}/logs/results.log" 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/router.sh" \
#   >> "${PROJECT_ROOT}/logs/router.log" 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/data.sh" \
#   >> "${PROJECT_ROOT}/logs/data.log" 2>&1 &
```

Now run the script.sh using below command:
```bash
./script.sh
```

Expected output:

```text
data/Classified/*.jsonl
logs/classifier.log
```

Each output JSONL file should contain one classified row per input prompt, with the original prompt metadata and a nested `classifier` result.

### 5.2 Dataset statistics by k

Run dataset analysis wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 nohup ./scripts/data.sh \
  >> "$PROJECT_ROOT/logs/result_analysis.log" 2>&1 &
```

Expected outputs include:

```text
data_stats/combined_cleaned_all_k.csv
results/dataset_stats_by_k/dataset_stats_by_k_report.md
results/dataset_stats_by_k/tables/overall_summary_by_k.csv
results/dataset_stats_by_k/tables/cell_balance_summary_by_k.csv
results/dataset_stats_by_k/tables/nestedness_adjacent_k.csv
results/dataset_stats_by_k/plots/*.png
```

These outputs support dataset-scale, balance, nestedness, and quality-control claims.

### 5.3 K-aware result analysis

Run the main result-analysis wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 nohup ./scripts/results.sh \
  >> "$PROJECT_ROOT/logs/result_analysis.log" 2>&1 &
```

Expected key outputs:

```text
results/k_aware_analysis/analysis_ready_flat.csv
results/k_aware_analysis/tables/01_model_metrics_by_k.csv
results/k_aware_analysis/tables/02_model_metrics_by_k_common_prompts.csv
results/k_aware_analysis/tables/03_model_k_coverage_summary.csv
results/k_aware_analysis/tables/05_resource_tier_metrics_by_k_model.csv
results/k_aware_analysis/tables/06_category_metrics_by_k_model.csv
results/k_aware_analysis/tables/07_language_metrics_by_k_model.csv
results/k_aware_analysis/tables/11_normalization_audit.csv
results/k_aware_analysis/figures/heatmap_evasion_model_by_k.png
results/k_aware_analysis/figures/heatmap_unsafe_recall_by_category_k2.png
results/k_aware_analysis/figures/heatmap_evasion_by_resource_tier_k2.png
results/k_aware_analysis/figures/line_error_rate_by_k.png
results/k_aware_analysis/figures/line_unknown_rate_by_k.png
```

The file `analysis_ready_flat.csv` (this file is not uploaded due to large size, the code (**k_aware_result_analysis.py**) will generate this file) is the input to the LinguaGuard routing experiment.

### 5.4 LinguaGuard router and ensemble evaluation

Run the router wrapper:

```bash
CUDA_VISIBLE_DEVICES=3 nohup "${PROJECT_ROOT}/scripts/router.sh" \
  >> "${PROJECT_ROOT}/logs/router.log" 2>&1 &
```

Expected key outputs:

```text
results/linguaguard_router_all_k/tables/00_model_availability_by_k.csv
results/linguaguard_router_all_k/tables/03_linguaguard_top3_global_all_k.csv
results/linguaguard_router_all_k/tables/04_linguaguard_top3_per_resource_tier_all_k.csv
results/linguaguard_router_all_k/tables/05_linguaguard_top3_per_language_all_k.csv
results/linguaguard_router_all_k/tables/07_linguaguard_main_comparison_paper_all_k.csv
results/linguaguard_router_all_k/tables/08_linguaguard_per_language_improvement_all_k.csv
results/linguaguard_router_all_k/tables/09_linguaguard_language_improvement_summary_all_k.csv
results/linguaguard_router_all_k/figures/linguaguard_strict_evasion_by_method_all_k.png
results/linguaguard_router_all_k/audit/linguaguard_test_predictions_audit_all_k.csv
```

The router evaluates the following methods separately for each `k` subset:

```text
Best single global
Language top-3 majority
Top-1 language router
Resource-tier top-3 OR
Global top-3 OR
LinguaGuard top-3 language OR
All-model OR upper bound
```

---

## 6. Paper Figure/Table to Artifact output mapping

The following mapping connects the numbered tables and figures in the paper to artifact outputs.

| Paper item | Paper location | Artifact output |
|---|---|---|
| Table 1: Guardrail model variants evaluated in GuardBreach | Section 4.5 | This is a manuscript summary table. The evaluated classifier keys are implemented in `src/multi_guard_prompt_classifier.py`, and generated outputs are saved under `data/classified/<model_name>/<model_name>_dataset_k=<k>.jsonl`. |
| Figure 1: Threat model for multilingual runtime guardrail evasion | Section 3 | Manuscript figure. Not generated by the result-analysis scripts. |
| Figure 2: GuardBreach methodology workflow | Section 4 | Manuscript figure. Not generated by the result-analysis scripts. |
| Figure 3: Unsafe recall by harm category and model for `k=1` | Section 5.1 | `results/k_aware_analysis/figures/heatmap_unsafe_recall_by_category_k1.png`; source table: `results/k_aware_analysis/tables/06_category_metrics_by_k_model.csv` |
| Figure 4: Strict evasion by language-resource tier and model for `k=1` | Section 5.2 | `results/k_aware_analysis/figures/heatmap_evasion_by_resource_tier_k1.png`; source table: `results/k_aware_analysis/tables/05_resource_tier_metrics_by_k_model.csv` |
| Figure 5: Strict evasion by language-resource tier and model for `k=2` | Section 5.2 | `results/k_aware_analysis/figures/heatmap_evasion_by_resource_tier_k2.png`; source table: `results/k_aware_analysis/tables/05_resource_tier_metrics_by_k_model.csv` |
| Figure 6: Strict evasion by language-resource tier and model for `k=3` | Section 5.2 | `results/k_aware_analysis/figures/heatmap_evasion_by_resource_tier_k3.png`; source table: `results/k_aware_analysis/tables/05_resource_tier_metrics_by_k_model.csv` |
| Table 2: Deployment-aware guardrail performance across GuardBreach subsets | Section 5.3 | `results/k_aware_analysis/tables/01_model_metrics_by_k.csv`; coverage/completion support: `results/k_aware_analysis/tables/03_model_k_coverage_summary.csv` and `results/k_aware_analysis/tables/04_model_k_completion_matrix.csv` |
| Figure 7: Strict evasion rate by LinguaGuard mitigation method across `k=1`, `k=2`, and `k=3` | Section 5.4 | `results/linguaguard_router_all_k/figures/linguaguard_strict_evasion_by_method_all_k.png`; source table: `results/linguaguard_router_all_k/tables/07_linguaguard_main_comparison_paper_all_k.csv` |
| Figure 8: Unsafe recall by harm category and model for `k=2` | Appendix B | `results/k_aware_analysis/figures/heatmap_unsafe_recall_by_category_k2.png`; source table: `results/k_aware_analysis/tables/06_category_metrics_by_k_model.csv` |
| Figure 9: Unsafe recall by harm category and model for `k=3` | Appendix B | `results/k_aware_analysis/figures/heatmap_unsafe_recall_by_category_k3.png`; source table: `results/k_aware_analysis/tables/06_category_metrics_by_k_model.csv` |
| Figure 10: Radar summary of top guardrail deployment profiles across GuardBreach subsets | Appendix C | `results/k_aware_analysis/figures/radar_top_models_k1.png`, `results/k_aware_analysis/figures/radar_top_models_k2.png`, and `results/k_aware_analysis/figures/radar_top_models_k3.png`; source table: `results/k_aware_analysis/tables/01_model_metrics_by_k.csv` |
| Figure 11: Pairwise model prediction agreement for `k=3` | Appendix D | `results/k_aware_analysis/figures/heatmap_model_agreement_k3.png`; generated from `results/k_aware_analysis/analysis_ready_flat.csv` |
| Figure 12: Pairwise model prediction agreement for `k=1` | Appendix D | `results/k_aware_analysis/figures/heatmap_model_agreement_k1.png`; generated from `results/k_aware_analysis/analysis_ready_flat.csv` |
| Figure 13: Pairwise model prediction agreement for `k=2` | Appendix D | `results/k_aware_analysis/figures/heatmap_model_agreement_k2.png`; generated from `results/k_aware_analysis/analysis_ready_flat.csv` |

(**Optional to see the below mapping**) The following supporting outputs are used for paper claims but are not currently assigned separate numbered tables or figures in the draft.

| Paper claim / analysis | Paper location | Artifact output |
|---|---|---|
| Strict evasion by model and `k` | Supports Table 2 and Section 5.3 | `results/k_aware_analysis/figures/heatmap_evasion_model_by_k.png` |
| Unsafe recall by model and `k` | Supports Table 2 and Section 5.3 | `results/k_aware_analysis/figures/heatmap_unsafe_recall_model_by_k.png` and `results/k_aware_analysis/figures/line_unsafe_recall_by_k.png` |
| Error and unknown-output behavior | Supports Section 5.3 | `results/k_aware_analysis/figures/line_error_rate_by_k.png`, `results/k_aware_analysis/figures/line_unknown_rate_by_k.png`, and `results/k_aware_analysis/tables/03_model_k_coverage_summary.csv` |
| Normalization audit | Supports Section 4.6 | `results/k_aware_analysis/tables/11_normalization_audit.csv` |
| Language-level rankings | Supports Section 4.9 and appendix analysis | `results/k_aware_analysis/tables/12_per_language_model_scores_by_k.csv`, `results/k_aware_analysis/tables/13_top10_models_per_language_by_k.csv`, and `results/k_aware_analysis/tables/14_best_model_per_language_by_k.csv` |
| Best-model language win counts | Supports language-level analysis | `results/k_aware_analysis/figures/bar_best_model_win_counts_k1.png`, `results/k_aware_analysis/figures/bar_best_model_win_counts_k2.png`, `results/k_aware_analysis/figures/bar_best_model_win_counts_k3.png`, and `results/k_aware_analysis/tables/15_model_language_win_counts_by_k.csv` |
| Translation-quality effects | Supports Sections 4.2, 4.6, and limitations | `results/k_aware_analysis/tables/08_quality_bucket_metrics_by_k_model.csv`, `results/k_aware_analysis/tables/17_quality_detection_correlations_by_k.csv`, and `results/k_aware_analysis/figures/line_combined_score_bin_evasion_by_k.png` |
| Prompt-length sensitivity | Supporting robustness analysis | `results/k_aware_analysis/figures/line_prompt_len_bin_evasion_by_k.png` |
| LinguaGuard selected global top-3 models | Supports Section 5.4 | `results/linguaguard_router_all_k/tables/03_linguaguard_top3_global_all_k.csv` |
| LinguaGuard selected resource-tier top-3 models | Supports Section 5.4 | `results/linguaguard_router_all_k/tables/04_linguaguard_top3_per_resource_tier_all_k.csv` |
| LinguaGuard selected per-language top-3 models | Supports Section 5.4 | `results/linguaguard_router_all_k/tables/05_linguaguard_top3_per_language_all_k.csv` |
| LinguaGuard per-language improvement | Supports Section 5.4 and appendix analysis | `results/linguaguard_router_all_k/tables/08_linguaguard_per_language_improvement_all_k.csv` and `results/linguaguard_router_all_k/tables/09_linguaguard_language_improvement_summary_all_k.csv` |
| LinguaGuard prediction audit | Reproducibility audit | `results/linguaguard_router_all_k/audit/linguaguard_test_predictions_audit_all_k.csv` |

---

## 7. Interpreting key metrics

The benchmark currently evaluates harmful prompts. Therefore, the primary security metrics are:

- `unsafe_recall_strict`: fraction of harmful prompts explicitly classified as `unsafe`.
- `evasion_rate_strict`: fraction of harmful prompts classified as anything other than `unsafe`.
- `coverage_rate`: fraction of rows with a valid `safe` or `unsafe` output.
- `unknown_rate`: fraction of rows normalized as `unknown`.
- `error_rate`: fraction of rows with runtime or execution failure.

Because the evaluated subsets contain unsafe prompts only, false-positive rate, specificity, over-refusal rate, and balanced accuracy are undefined unless a benign multilingual subset is added.

---

## 8. Notes on reproducibility and expected runtime

Full classifier inference can be expensive because it evaluates multiple guardrail models across up to three benchmark subsets. Runtime depends on GPU type, model access, model size, and whether the model weights are already cached. To support review under limited compute, the artifact should include precomputed `data/Classified/*.jsonl` outputs and the downstream analysis outputs whenever licensing and safety restrictions permit.

Recommended reviewer workflow under limited compute:

1. Inspect dataset schemas and sample rows.
2. Run `dataset_stats_by_k.py` on the provided benchmark files.
3. Run `k_aware_result_analysis.py` on provided `data/classified/*.jsonl` files.
4. Run `linguaguard_router_all_k.py` on the generated `analysis_ready_flat.csv` (this file is not uploaded because of size, it will be generated by the code).
5. Optionally rerun a small classifier subset for one model and one `k` file to verify functionality.
---

## 10. Current artifact limitations

- Some model weights may require external access approval or acceptance of model licenses.
- Some prompt text or model outputs may require restricted release due to safety or licensing concerns.
- `k=3` is a partial stress test and should not be interpreted as a complete all-model comparison.

---
