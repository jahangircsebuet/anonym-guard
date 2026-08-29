# GuardBreach Dataset Insights

This folder contains 8 paper-ready dataset insight groups.

Important: this script treats the dataset column `tier` as **language resource tier**, not harm/severity tier.
The script also creates a clearer alias column named `language_resource_tier`.

## Insight 1: Dataset scale by k
- `tables/01_dataset_scale_by_k.csv`
- `plots/01_prompts_across_k.png`
- `plots/01_unique_roots_across_k.png`

## Insight 2: Language coverage
- `tables/02_language_counts_by_k.csv`
- `tables/02_language_percent_by_k.csv`
- `tables/02_language_summary.csv`
- `plots/02_top_languages_by_count.png`
- `plots/02_language_by_k_heatmap.png`

## Insight 3: Language resource-tier distribution
Resource-tier column used: `language_resource_tier`

- `tables/03_language_resource_tier_counts_by_k.csv`
- `tables/03_language_resource_tier_percent_by_k.csv`
- `tables/03_language_resource_tier_summary_by_k.csv`
- `plots/03_language_resource_tier_counts_by_k.png`

## Insight 4: Harm-category coverage
- `tables/04_category_counts_by_k.csv`
- `tables/04_category_percent_by_k.csv`
- `tables/04_category_summary.csv`
- `plots/04_category_counts_by_k.png`

## Insight 5: Category x language-resource-tier balance
- `tables/05_category_language_resource_tier_counts_long.csv`
- `tables/05_category_x_language_resource_tier_counts_k1.csv`
- `tables/05_category_x_language_resource_tier_counts_k2.csv`
- `tables/05_category_x_language_resource_tier_counts_k3.csv`
- `plots/05_category_x_language_resource_tier_heatmap_k1.png`
- `plots/05_category_x_language_resource_tier_heatmap_k2.png`
- `plots/05_category_x_language_resource_tier_heatmap_k3.png`

## Insight 6: Language x category cell balance
Because `tier` is language resource tier, language x category x tier is not used for missing-cell analysis.
Each language belongs to one resource tier, so the meaningful stratification check is language x category.

- `tables/06_full_language_category_cell_counts_by_k.csv`
- `tables/06_missing_language_category_cells_by_k.csv`
- `tables/06_language_category_cell_balance_summary_by_k.csv`
- `tables/06_resource_tier_category_counts_by_k.csv`
- `plots/06_observed_language_category_cells_across_k.png`
- `plots/06_missing_language_category_cells_across_k.png`
- `plots/06_resource_tier_x_category_heatmap_k1.png`
- `plots/06_resource_tier_x_category_heatmap_k2.png`
- `plots/06_resource_tier_x_category_heatmap_k3.png`

## Insight 7: Translation-quality distribution
- `tables/07_translation_quality_summary_by_k.csv`
- `tables/07_quality_bucket_counts_by_k.csv`
- `tables/07_quality_bucket_percent_by_k.csv`
- `plots/07_merged_boxplot_f1_comet_combined_score_by_k.png`
- `plots/07_boxplot_selection_score_by_k.png`
- `plots/07_quality_bucket_counts_by_k.png`

## Insight 8: Prompt-length distribution
- `tables/08_prompt_length_summary_by_k.csv`
- `tables/08_prompt_length_by_language_and_k.csv`
- `tables/08_prompt_length_by_category_and_k.csv`
- `tables/08_prompt_length_by_language_resource_tier_and_k.csv`
- `plots/08_boxplot_prompt_chars_by_k.png`
- `plots/08_hist_prompt_chars_by_k.png`
