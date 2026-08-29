## Translation-Quality Filtering and Quality Buckets

Before constructing the final GuardBreach benchmark subsets, we score translated prompts using both BERTScore and COMET. The filtering script merges the BERTScore table and COMET table using the shared keys:

```text
root_id
language
```

The script then computes a combined translation-quality score:

```text
combined_score = (BERTScore_F1 + COMET) / 2      if COMET is available
combined_score = BERTScore_F1                    otherwise
```

It also records whether COMET was available:

```text
comet_available = 1 if COMET exists
comet_available = 0 otherwise
```

### Quality Bucket Definitions

Each translated prompt is assigned to exactly one disjoint quality bucket:

```text
gold ∩ high_quality ∩ standard_quality ∩ low_quality = empty
```

The thresholds are:

| Bucket | BERTScore F1 | COMET | Combined score |
|---|---:|---:|---:|
| `gold` | ≥ 0.95 | ≥ 0.92 | ≥ 0.94 |
| `high_quality` | ≥ 0.90 | ≥ 0.85 | ≥ 0.88 |
| `standard_quality` | ≥ 0.80 | ≥ 0.70 | ≥ 0.75 |
| `low_quality` | otherwise | otherwise | otherwise |

When COMET is unavailable, the script falls back to BERTScore F1 only:

| Bucket | BERTScore-only fallback |
|---|---:|
| `gold` | F1 ≥ 0.95 |
| `high_quality` | F1 ≥ 0.90 |
| `standard_quality` | F1 ≥ 0.80 |
| `low_quality` | F1 < 0.80 |

The bucket assignment is hierarchical and disjoint. A prompt that passes the `gold` threshold is assigned only to `gold`; it is not also counted as `high_quality` or `standard_quality`.

### Best Translation per Root-Language Pair

In addition to quality buckets, the script creates a best-per-root-language file. For each pair:

```text
(root_id, language)
```

the script keeps the row with the highest `combined_score`. This produces:

```text
sampled_best_per_root_language.csv
```

This file is useful when we need one highest-quality translated prompt for each source prompt and language.

### Running the Quality Filtering Script

Run the script as follows:

```bash
python "${PROJECT_ROOT}/src/filter_translation_quality.py" \
  --bert_file "${PROJECT_ROOT}/data/quality/bertscore/evaluation_table.csv" \
  --comet_file "${PROJECT_ROOT}/data/quality/comet/evaluation_table_comet_full.csv" \
  --output_dir "${PROJECT_ROOT}/results/translation_quality_filtering"
```

### Outputs

The script writes the following files:

```text
results/translation_quality_filtering/
  merged_scores.csv
  sampled_gold.csv
  sampled_high_quality.csv
  sampled_standard_quality.csv
  sampled_low_quality.csv
  sampled_best_per_root_language.csv
  sampled_summary.json
```

The summary file records the number and percentage of rows assigned to each bucket, the bucket definitions, and the exact thresholds used for filtering.

### Role in GuardBreach

The resulting quality metadata is carried into the GuardBreach benchmark records through fields such as:

```text
f1
comet
combined_score
comet_available
quality_bucket
selection_score
```

These fields are later used for stratified sampling, quality-aware analysis, and checking whether multilingual guardrail evasion persists even for high-quality translations.