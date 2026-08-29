#!/usr/bin/env python3
"""
Dataset statistics for GuardBreach sampled benchmark subsets across k values.

This script expects JSONL files where each row looks like:

{
  "root_id": "...",
  "language": "Zulu",
  "category": "Harassment",
  "tier": "low",
  "f1": "0.8993",
  "prompt_len": "33",
  "label": "unsafe",
  "failure": "0",
  "severe_failure": "0",
  "comet": "",
  "combined_score": "0.8993",
  "comet_available": "0",
  "quality_bucket": "standard_quality",
  "selection_score": "0.8993",
  "sampling_strategy": "stratified_top_k",
  "k_per_cell": "1",
  "prompt": "..."
}

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DEFAULT_FILES = {
    1: str(PROJECT_ROOT / "data/stratified/stratified_topk_k1_with_prompts.jsonl"),
    2: str(PROJECT_ROOT / "data/stratified/stratified_topk_k2_with_prompts.jsonl"),
    3: str(PROJECT_ROOT / "data/stratified/stratified_topk_k3_with_prompts.jsonl"),
}

EXPECTED_COLS = [
    "root_id",
    "language",
    "category",
    "tier",
    "f1",
    "prompt_len",
    "label",
    "failure",
    "severe_failure",
    "comet",
    "combined_score",
    "comet_available",
    "quality_bucket",
    "selection_score",
    "sampling_strategy",
    "k_per_cell",
    "prompt",
]


NUMERIC_COLS = [
    "f1",
    "prompt_len",
    "failure",
    "severe_failure",
    "comet",
    "combined_score",
    "comet_available",
    "selection_score",
    "k_per_cell",
]


GROUP_COLS = [
    "language",
    "category",
    "tier",
    "label",
    "quality_bucket",
    "sampling_strategy",
]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    bad_lines = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines.append((line_no, str(e), line[:200]))

    if bad_lines:
        print(f"[warning] {path} has {len(bad_lines)} malformed lines.")
        for line_no, err, preview in bad_lines[:5]:
            print(f"  line={line_no}, error={err}, preview={preview}")

    return pd.DataFrame(rows)


def infer_k_from_path(path: str) -> int:
    """
    Infer k from filename pattern such as:
    benchmark_strategy1_topk_k3_with_prompts.jsonl
    """
    name = Path(path).name
    match = re.search(r"_k(\d+)_", name)
    if match:
        return int(match.group(1))

    match = re.search(r"k(\d+)", name)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not infer k from path: {path}")


def clean_dataframe(df: pd.DataFrame, k_value: int, source_path: str) -> pd.DataFrame:
    df = df.copy()

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    for col in GROUP_COLS:
        df[col] = df[col].astype("string").str.strip()

    df["root_id"] = df["root_id"].astype("string").str.strip()
    df["prompt"] = df["prompt"].fillna("").astype(str)

    df["k_subset"] = int(k_value)
    # df["source_file"] = source_path
    df["source_file"] = Path(source_path).name

    df["prompt_char_len_actual"] = df["prompt"].str.len()
    df["prompt_word_len_rough"] = df["prompt"].str.split().str.len()

    # Stable row key for overlap/nestedness checks.
    df["row_key"] = (
        df["root_id"].fillna("NA").astype(str)
        + "||"
        + df["language"].fillna("NA").astype(str)
        + "||"
        + df["category"].fillna("NA").astype(str)
        + "||"
        + df["tier"].fillna("NA").astype(str)
        + "||"
        + df["label"].fillna("NA").astype(str)
        + "||"
        + df["prompt"].fillna("").astype(str)
    )

    return df


def load_all_k(files: Dict[int, str]) -> pd.DataFrame:
    frames = []

    for k, path_str in sorted(files.items()):
        path = Path(path_str)

        if not path.exists():
            print(f"[warning] Missing file for k={k}: {path}")
            continue

        print(f"[load] k={k}: {path}")
        df = load_jsonl(path)
        df = clean_dataframe(df, k_value=k, source_path=str(path))
        frames.append(df)

    if not frames:
        raise RuntimeError("No dataset files were loaded. Check file paths.")

    return pd.concat(frames, ignore_index=True)


def save_table(df: pd.DataFrame, path: Path, index: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")


def value_counts_by_k(df: pd.DataFrame, col: str) -> pd.DataFrame:
    table = (
        df.groupby(["k_subset", col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    totals = table.groupby("k_subset")["count"].transform("sum")
    table["percent_within_k"] = 100.0 * table["count"] / totals

    return table.sort_values(["k_subset", "count"], ascending=[True, False])


def pivot_counts_by_k(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.crosstab(df[col], df["k_subset"], margins=True, dropna=False)


def group_rate_by_k(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = (
        df.groupby(["k_subset", group_col], dropna=False)
        .agg(
            n=("prompt", "size"),
            unique_roots=("root_id", "nunique"),
            unsafe_rate=("label", lambda x: np.mean(x == "unsafe")),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
            avg_f1=("f1", "mean"),
            avg_comet=("comet", "mean"),
            avg_combined_score=("combined_score", "mean"),
            avg_selection_score=("selection_score", "mean"),
            avg_prompt_chars=("prompt_char_len_actual", "mean"),
            avg_prompt_words=("prompt_word_len_rough", "mean"),
        )
        .reset_index()
        .sort_values(["k_subset", "failure_rate", "n"], ascending=[True, False, False])
    )
    return table


def overall_summary_by_k(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("k_subset")
        .agg(
            rows=("prompt", "size"),
            unique_row_keys=("row_key", "nunique"),
            unique_roots=("root_id", "nunique"),
            unique_languages=("language", "nunique"),
            unique_categories=("category", "nunique"),
            unique_tiers=("tier", "nunique"),
            unique_labels=("label", "nunique"),
            unsafe_rate=("label", lambda x: np.mean(x == "unsafe")),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
            avg_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            avg_comet=("comet", "mean"),
            median_comet=("comet", "median"),
            avg_combined_score=("combined_score", "mean"),
            median_combined_score=("combined_score", "median"),
            avg_selection_score=("selection_score", "mean"),
            avg_prompt_chars=("prompt_char_len_actual", "mean"),
            median_prompt_chars=("prompt_char_len_actual", "median"),
            avg_prompt_words=("prompt_word_len_rough", "mean"),
            median_prompt_words=("prompt_word_len_rough", "median"),
            comet_available_rate=("comet_available", "mean"),
        )
        .reset_index()
        .sort_values("k_subset")
    )

    return summary


def numeric_summary_by_k(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "f1",
        "comet",
        "combined_score",
        "selection_score",
        "prompt_len",
        "prompt_char_len_actual",
        "prompt_word_len_rough",
        "failure",
        "severe_failure",
    ]

    rows = []

    for k, g in df.groupby("k_subset"):
        for col in numeric_cols:
            if col not in g.columns:
                continue

            s = g[col].dropna()

            if len(s) == 0:
                continue

            rows.append(
                {
                    "k_subset": k,
                    "metric": col,
                    "count": len(s),
                    "mean": s.mean(),
                    "std": s.std(),
                    "min": s.min(),
                    "p01": s.quantile(0.01),
                    "p05": s.quantile(0.05),
                    "p25": s.quantile(0.25),
                    "median": s.quantile(0.50),
                    "p75": s.quantile(0.75),
                    "p95": s.quantile(0.95),
                    "p99": s.quantile(0.99),
                    "max": s.max(),
                }
            )

    return pd.DataFrame(rows)


def cell_balance_by_k(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes language x category x tier cell count for each k.

    Useful for checking whether the top-k sampling is balanced.
    """
    cell_counts = (
        df.groupby(["k_subset", "language", "category", "tier"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["k_subset", "language", "category", "tier"])
    )

    cell_summary = (
        cell_counts.groupby("k_subset")
        .agg(
            observed_cells=("count", "size"),
            min_count_per_observed_cell=("count", "min"),
            max_count_per_observed_cell=("count", "max"),
            mean_count_per_observed_cell=("count", "mean"),
            median_count_per_observed_cell=("count", "median"),
            std_count_per_observed_cell=("count", "std"),
            cells_with_count_1=("count", lambda x: int((x == 1).sum())),
            cells_with_count_2=("count", lambda x: int((x == 2).sum())),
            cells_with_count_3=("count", lambda x: int((x == 3).sum())),
            cells_with_count_4=("count", lambda x: int((x == 4).sum())),
            cells_with_count_5=("count", lambda x: int((x == 5).sum())),
        )
        .reset_index()
        .sort_values("k_subset")
    )

    # Full universe based on union across all loaded k files.
    languages = sorted(df["language"].dropna().unique())
    categories = sorted(df["category"].dropna().unique())
    tiers = sorted(df["tier"].dropna().unique())
    ks = sorted(df["k_subset"].dropna().unique())

    full_index = pd.MultiIndex.from_product(
        [ks, languages, categories, tiers],
        names=["k_subset", "language", "category", "tier"],
    )

    full_cell_counts = (
        cell_counts.set_index(["k_subset", "language", "category", "tier"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return cell_counts, full_cell_counts, cell_summary


def root_coverage_by_k(df: pd.DataFrame) -> pd.DataFrame:
    root_cov = (
        df.groupby(["k_subset", "root_id"], dropna=False)
        .agg(
            n_rows=("prompt", "size"),
            n_languages=("language", "nunique"),
            n_categories=("category", "nunique"),
            n_tiers=("tier", "nunique"),
            n_labels=("label", "nunique"),
            avg_f1=("f1", "mean"),
            avg_comet=("comet", "mean"),
            avg_combined_score=("combined_score", "mean"),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
        )
        .reset_index()
        .sort_values(["k_subset", "n_languages", "failure_rate"], ascending=[True, True, False])
    )
    return root_cov


def overlap_and_nestedness(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes pairwise overlap between k subsets.

    This helps verify whether k=1 is contained inside k=2, k=2 inside k=3, etc.
    """
    key_sets = {
        int(k): set(g["row_key"].dropna().astype(str).tolist())
        for k, g in df.groupby("k_subset")
    }

    rows = []

    for k_a in sorted(key_sets):
        for k_b in sorted(key_sets):
            a = key_sets[k_a]
            b = key_sets[k_b]

            inter = a.intersection(b)
            union = a.union(b)

            rows.append(
                {
                    "k_a": k_a,
                    "k_b": k_b,
                    "n_a": len(a),
                    "n_b": len(b),
                    "intersection": len(inter),
                    "union": len(union),
                    "jaccard": len(inter) / len(union) if union else np.nan,
                    "percent_of_a_in_b": 100.0 * len(inter) / len(a) if a else np.nan,
                    "percent_of_b_in_a": 100.0 * len(inter) / len(b) if b else np.nan,
                }
            )

    pairwise = pd.DataFrame(rows)

    nested_rows = []
    sorted_ks = sorted(key_sets)

    for prev_k, next_k in zip(sorted_ks[:-1], sorted_ks[1:]):
        prev_set = key_sets[prev_k]
        next_set = key_sets[next_k]

        missing_from_next = prev_set - next_set
        newly_added = next_set - prev_set

        nested_rows.append(
            {
                "from_k": prev_k,
                "to_k": next_k,
                "rows_in_from_k": len(prev_set),
                "rows_in_to_k": len(next_set),
                "from_k_rows_found_in_to_k": len(prev_set.intersection(next_set)),
                "from_k_rows_missing_in_to_k": len(missing_from_next),
                "percent_from_k_preserved": 100.0
                * len(prev_set.intersection(next_set))
                / len(prev_set)
                if prev_set
                else np.nan,
                "new_rows_added_in_to_k": len(newly_added),
            }
        )

    nestedness = pd.DataFrame(nested_rows)

    return pairwise, nestedness


def incremental_rows_by_k(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies rows newly appearing as k increases.

    For top-k nested sampling, k=2 should contain all k=1 rows plus one extra row
    per eligible cell, etc.
    """
    key_to_first_k = (
        df.groupby("row_key")["k_subset"]
        .min()
        .rename("first_seen_k")
        .reset_index()
    )

    tmp = df.merge(key_to_first_k, on="row_key", how="left")

    inc = (
        tmp[tmp["k_subset"] == tmp["first_seen_k"]]
        .groupby("first_seen_k")
        .agg(
            newly_seen_rows=("row_key", "nunique"),
            newly_seen_roots=("root_id", "nunique"),
            newly_seen_languages=("language", "nunique"),
            newly_seen_categories=("category", "nunique"),
            avg_f1=("f1", "mean"),
            avg_comet=("comet", "mean"),
            avg_combined_score=("combined_score", "mean"),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
        )
        .reset_index()
        .rename(columns={"first_seen_k": "k_subset"})
        .sort_values("k_subset")
    )

    return inc


def quality_flags_by_k(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    flags["missing_prompt"] = df["prompt"].isna() | (df["prompt"].str.len() == 0)
    flags["missing_language"] = df["language"].isna()
    flags["missing_category"] = df["category"].isna()
    flags["missing_label"] = df["label"].isna()
    flags["missing_f1"] = df["f1"].isna()
    flags["missing_comet_when_available"] = (df["comet_available"] == 1) & df["comet"].isna()
    flags["low_f1_lt_0_80"] = df["f1"] < 0.80
    flags["low_comet_lt_0_70"] = df["comet"] < 0.70
    flags["low_combined_lt_0_80"] = df["combined_score"] < 0.80
    flags["failure_flag"] = df["failure"] == 1
    flags["severe_failure_flag"] = df["severe_failure"] == 1
    flags["very_short_prompt_lt_10_chars"] = df["prompt_char_len_actual"] < 10
    flags["very_long_prompt_gt_1000_chars"] = df["prompt_char_len_actual"] > 1000
    flags["prompt_len_mismatch"] = (
        df["prompt_len"].notna()
        & (df["prompt_len"] != df["prompt_char_len_actual"])
    )

    out = df.copy()
    out["num_quality_flags"] = flags.sum(axis=1)

    for col in flags.columns:
        out[col] = flags[col]

    flagged_rows = out[out["num_quality_flags"] > 0].sort_values(
        ["k_subset", "num_quality_flags", "combined_score"],
        ascending=[True, False, True],
    )

    flag_summary = (
        out.groupby("k_subset")
        .agg(
            rows=("prompt", "size"),
            rows_with_any_quality_flag=("num_quality_flags", lambda x: int((x > 0).sum())),
            avg_quality_flags_per_row=("num_quality_flags", "mean"),
            missing_prompt=("missing_prompt", "sum"),
            missing_language=("missing_language", "sum"),
            missing_category=("missing_category", "sum"),
            missing_label=("missing_label", "sum"),
            missing_f1=("missing_f1", "sum"),
            missing_comet_when_available=("missing_comet_when_available", "sum"),
            low_f1_lt_0_80=("low_f1_lt_0_80", "sum"),
            low_comet_lt_0_70=("low_comet_lt_0_70", "sum"),
            low_combined_lt_0_80=("low_combined_lt_0_80", "sum"),
            failure_flag=("failure_flag", "sum"),
            severe_failure_flag=("severe_failure_flag", "sum"),
            very_short_prompt_lt_10_chars=("very_short_prompt_lt_10_chars", "sum"),
            very_long_prompt_gt_1000_chars=("very_long_prompt_gt_1000_chars", "sum"),
            prompt_len_mismatch=("prompt_len_mismatch", "sum"),
        )
        .reset_index()
    )

    flag_summary["percent_rows_with_any_quality_flag"] = (
        100.0 * flag_summary["rows_with_any_quality_flag"] / flag_summary["rows"]
    )

    return flagged_rows, flag_summary


def duplicate_summary_by_k(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for k, g in df.groupby("k_subset"):
        duplicate_exact_prompt_rows = int(g.duplicated(subset=["prompt"], keep=False).sum())
        duplicate_root_language_rows = int(
            g.duplicated(subset=["root_id", "language"], keep=False).sum()
        )
        duplicate_root_language_category_tier_rows = int(
            g.duplicated(
                subset=["root_id", "language", "category", "tier"], keep=False
            ).sum()
        )

        rows.append(
            {
                "k_subset": k,
                "rows": len(g),
                "duplicate_exact_prompt_rows": duplicate_exact_prompt_rows,
                "duplicate_root_language_rows": duplicate_root_language_rows,
                "duplicate_root_language_category_tier_rows": duplicate_root_language_category_tier_rows,
                "percent_duplicate_exact_prompt_rows": 100.0
                * duplicate_exact_prompt_rows
                / len(g)
                if len(g)
                else np.nan,
                "percent_duplicate_root_language_rows": 100.0
                * duplicate_root_language_rows
                / len(g)
                if len(g)
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("k_subset")


def plot_line_metric(summary: pd.DataFrame, metric: str, outpath: Path, ylabel: str = None):
    plt.figure(figsize=(8, 5))
    plt.plot(summary["k_subset"], summary[metric], marker="o")
    plt.title(f"{metric} across k")
    plt.xlabel("k per cell")
    plt.ylabel(ylabel or metric)
    plt.xticks(summary["k_subset"].tolist())
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_distribution_by_k(df: pd.DataFrame, col: str, outpath: Path):
    table = pd.crosstab(df[col], df["k_subset"], normalize="columns") * 100.0

    ax = table.plot(kind="bar", figsize=(12, 6))
    ax.set_title(f"{col} distribution across k")
    ax.set_xlabel(col)
    ax.set_ylabel("Percent within k")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_top_failure_by_k(
    group_stats: pd.DataFrame,
    group_col: str,
    outpath: Path,
    min_n: int = 10,
    top_n: int = 20,
):
    """
    For each group, average its failure rate across k values and plot top groups.
    """
    filtered = group_stats[group_stats["n"] >= min_n].copy()

    top_groups = (
        filtered.groupby(group_col)
        .agg(avg_failure_rate_across_k=("failure_rate", "mean"), total_n=("n", "sum"))
        .sort_values(["avg_failure_rate_across_k", "total_n"], ascending=[False, False])
        .head(top_n)
        .index
    )

    plot_df = filtered[filtered[group_col].isin(top_groups)]

    pivot = plot_df.pivot_table(
        index=group_col,
        columns="k_subset",
        values="failure_rate",
        aggfunc="mean",
    )

    pivot = pivot.loc[
        pivot.mean(axis=1).sort_values(ascending=False).index
    ]

    ax = pivot.plot(kind="bar", figsize=(14, 7))
    ax.set_title(f"Top {top_n} {group_col}s by failure rate across k")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Failure rate")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_markdown_report(
    df: pd.DataFrame,
    overall: pd.DataFrame,
    nestedness: pd.DataFrame,
    quality_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    outpath: Path,
):
    lines = []

    lines.append("# GuardBreach Dataset Statistics Across k\n")

    lines.append("## Loaded k Subsets\n")
    loaded = (
        df.groupby("k_subset")
        .agg(
            rows=("prompt", "size"),
            source_file=("source_file", "first"),
        )
        .reset_index()
        .sort_values("k_subset")
    )
    lines.append(loaded.to_markdown(index=False))
    lines.append("")

    lines.append("## Overall Summary by k\n")
    lines.append(overall.to_markdown(index=False))
    lines.append("")

    lines.append("## Nestedness Check\n")
    lines.append(
        "This checks whether rows in `k=1` are preserved in `k=2`, rows in `k=2` are preserved in `k=3`, etc."
    )
    lines.append("")
    lines.append(nestedness.to_markdown(index=False))
    lines.append("")

    lines.append("## Cell Balance Summary\n")
    lines.append(
        "This summarizes counts per `(language, category, tier)` cell for each k."
    )
    lines.append("")
    lines.append(cell_summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Quality Flag Summary\n")
    lines.append(quality_summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Recommended Tables for Paper\n")
    lines.append("- `tables/overall_summary_by_k.csv`")
    lines.append("- `tables/distribution_language_by_k.csv`")
    lines.append("- `tables/distribution_category_by_k.csv`")
    lines.append("- `tables/distribution_tier_by_k.csv`")
    lines.append("- `tables/group_stats_language_by_k.csv`")
    lines.append("- `tables/group_stats_category_by_k.csv`")
    lines.append("- `tables/group_stats_language_category_by_k.csv`")
    lines.append("- `tables/cell_counts_by_k_language_category_tier.csv`")
    lines.append("- `tables/nestedness_adjacent_k.csv`")
    lines.append("- `tables/incremental_rows_by_k.csv`")
    lines.append("- `tables/quality_flag_summary_by_k.csv`")
    lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def parse_file_args(file_args: List[str]) -> Dict[int, str]:
    """
    Allows either:
      --files /path/k1.jsonl /path/k2.jsonl
    or:
      --files 1=/path/k1.jsonl 2=/path/k2.jsonl
    """
    if not file_args:
        return DEFAULT_FILES

    files = {}

    for item in file_args:
        if "=" in item:
            k_str, path_str = item.split("=", 1)
            k = int(k_str)
        else:
            path_str = item
            k = infer_k_from_path(path_str)

        files[k] = path_str

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help=(
            "Optional list of JSONL files. "
            "Either pass paths directly or k=path pairs. "
            "Example: --files 1=/path/k1.jsonl 2=/path/k2.jsonl"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory",
    )
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--min_n_for_failure_plots", type=int, default=10)

    args = parser.parse_args()

    files = parse_file_args(args.files)
    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    per_k_dir = outdir / "per_k"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_k_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_k(files)

    # Save combined cleaned file.
    save_table(df, outdir / "combined_cleaned_all_k.csv", index=False)

    # Main summaries.
    overall = overall_summary_by_k(df)
    numeric_summary = numeric_summary_by_k(df)

    save_table(overall, tables_dir / "overall_summary_by_k.csv", index=False)
    save_table(numeric_summary, tables_dir / "numeric_summary_by_k.csv", index=False)

    # Distributions by k.
    for col in [
        "language",
        "category",
        "tier",
        "label",
        "quality_bucket",
        "sampling_strategy",
        "comet_available",
        "failure",
        "severe_failure",
    ]:
        dist = value_counts_by_k(df, col)
        save_table(dist, tables_dir / f"distribution_{col}_by_k.csv", index=False)

        pivot = pivot_counts_by_k(df, col)
        save_table(pivot, tables_dir / f"pivot_{col}_counts_by_k.csv")

    # Group stats by k.
    for group_col in ["language", "category", "tier", "label", "quality_bucket"]:
        stats = group_rate_by_k(df, group_col)
        save_table(stats, tables_dir / f"group_stats_{group_col}_by_k.csv", index=False)

    # Multi-dimensional group stats.
    lang_cat = (
        df.groupby(["k_subset", "language", "category"], dropna=False)
        .agg(
            n=("prompt", "size"),
            unique_roots=("root_id", "nunique"),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
            avg_f1=("f1", "mean"),
            avg_comet=("comet", "mean"),
            avg_combined_score=("combined_score", "mean"),
            avg_selection_score=("selection_score", "mean"),
            avg_prompt_chars=("prompt_char_len_actual", "mean"),
        )
        .reset_index()
        .sort_values(["k_subset", "failure_rate", "n"], ascending=[True, False, False])
    )
    save_table(lang_cat, tables_dir / "group_stats_language_category_by_k.csv", index=False)

    lang_tier = (
        df.groupby(["k_subset", "language", "tier"], dropna=False)
        .agg(
            n=("prompt", "size"),
            unique_roots=("root_id", "nunique"),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
            avg_f1=("f1", "mean"),
            avg_comet=("comet", "mean"),
            avg_combined_score=("combined_score", "mean"),
        )
        .reset_index()
        .sort_values(["k_subset", "failure_rate", "n"], ascending=[True, False, False])
    )
    save_table(lang_tier, tables_dir / "group_stats_language_tier_by_k.csv", index=False)

    cat_tier = (
        df.groupby(["k_subset", "category", "tier"], dropna=False)
        .agg(
            n=("prompt", "size"),
            unique_roots=("root_id", "nunique"),
            failure_rate=("failure", "mean"),
            severe_failure_rate=("severe_failure", "mean"),
            avg_f1=("f1", "mean"),
            avg_comet=("comet", "mean"),
            avg_combined_score=("combined_score", "mean"),
        )
        .reset_index()
        .sort_values(["k_subset", "failure_rate", "n"], ascending=[True, False, False])
    )
    save_table(cat_tier, tables_dir / "group_stats_category_tier_by_k.csv", index=False)

    # Cell balance.
    cell_counts, full_cell_counts, cell_summary = cell_balance_by_k(df)
    save_table(cell_counts, tables_dir / "cell_counts_by_k_language_category_tier.csv", index=False)
    save_table(full_cell_counts, tables_dir / "full_cell_counts_by_k_language_category_tier.csv", index=False)
    save_table(cell_summary, tables_dir / "cell_balance_summary_by_k.csv", index=False)

    missing_cells = full_cell_counts[full_cell_counts["count"] == 0]
    save_table(missing_cells, tables_dir / "missing_cells_by_k_language_category_tier.csv", index=False)

    # Root coverage.
    root_cov = root_coverage_by_k(df)
    save_table(root_cov, tables_dir / "root_coverage_by_k.csv", index=False)

    # Overlap and nestedness.
    pairwise_overlap, nestedness = overlap_and_nestedness(df)
    save_table(pairwise_overlap, tables_dir / "pairwise_overlap_by_k.csv", index=False)
    save_table(nestedness, tables_dir / "nestedness_adjacent_k.csv", index=False)

    # Incremental rows.
    inc = incremental_rows_by_k(df)
    save_table(inc, tables_dir / "incremental_rows_by_k.csv", index=False)

    # Duplicates.
    dups = duplicate_summary_by_k(df)
    save_table(dups, tables_dir / "duplicate_summary_by_k.csv", index=False)

    # Quality flags.
    flagged_rows, quality_summary = quality_flags_by_k(df)
    save_table(flagged_rows, tables_dir / "rows_needing_quality_inspection_by_k.csv", index=False)
    save_table(quality_summary, tables_dir / "quality_flag_summary_by_k.csv", index=False)

    # Per-k detailed reports.
    for k, g in df.groupby("k_subset"):
        k_dir = per_k_dir / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        save_table(g, k_dir / f"cleaned_k{k}.csv", index=False)

        for col in [
            "language",
            "category",
            "tier",
            "label",
            "quality_bucket",
            "comet_available",
            "failure",
            "severe_failure",
        ]:
            dist = value_counts_by_k(g, col)
            save_table(dist, k_dir / f"distribution_{col}_k{k}.csv", index=False)

        for group_col in ["language", "category", "tier", "quality_bucket"]:
            stats = group_rate_by_k(g, group_col)
            save_table(stats, k_dir / f"group_stats_{group_col}_k{k}.csv", index=False)

    # Plots: summary metrics across k.
    for metric in [
        "rows",
        "unique_roots",
        "unique_languages",
        "unique_categories",
        "failure_rate",
        "severe_failure_rate",
        "avg_f1",
        "avg_comet",
        "avg_combined_score",
        "avg_selection_score",
        "avg_prompt_chars",
        "comet_available_rate",
    ]:
        if metric in overall.columns:
            plot_line_metric(overall, metric, plots_dir / f"line_{metric}_across_k.png")

    # Distribution plots across k.
    for col in ["tier", "label", "quality_bucket", "failure", "severe_failure"]:
        plot_distribution_by_k(df, col, plots_dir / f"distribution_{col}_across_k.png")

    # Too many languages/categories can make plots hard to read, so only failure plots for top groups.
    lang_stats = group_rate_by_k(df, "language")
    cat_stats = group_rate_by_k(df, "category")

    plot_top_failure_by_k(
        lang_stats,
        "language",
        plots_dir / "top_languages_failure_rate_across_k.png",
        min_n=args.min_n_for_failure_plots,
        top_n=args.top_n,
    )

    plot_top_failure_by_k(
        cat_stats,
        "category",
        plots_dir / "top_categories_failure_rate_across_k.png",
        min_n=args.min_n_for_failure_plots,
        top_n=args.top_n,
    )

    # Histograms per k for quality metrics.
    for metric in ["f1", "comet", "combined_score", "selection_score", "prompt_char_len_actual"]:
        plt.figure(figsize=(10, 6))

        for k, g in df.groupby("k_subset"):
            values = g[metric].dropna()
            if len(values) == 0:
                continue
            plt.hist(values, bins=40, alpha=0.35, label=f"k={k}")

        plt.title(f"{metric} distribution across k")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"hist_{metric}_overlay_by_k.png", dpi=200)
        plt.close()

    # Markdown report.
    write_markdown_report(
        df=df,
        overall=overall,
        nestedness=nestedness,
        quality_summary=quality_summary,
        cell_summary=cell_summary,
        outpath=outdir / "dataset_stats_by_k_report.md",
    )

    print("\nDone.")
    print(f"Loaded rows across all k files: {len(df):,}")
    print(f"Loaded k values: {sorted(df['k_subset'].unique().tolist())}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Main report: {(outdir / 'dataset_stats_by_k_report.md').resolve()}")
    print(f"Main table: {(tables_dir / 'overall_summary_by_k.csv').resolve()}")


if __name__ == "__main__":
    main()