"""Aggregate ECGCLIP result JSON files across seeds.

Examples:
    python aggregate_results.py
    python aggregate_results.py --pattern 'work/results/biomedcoop_*.json'
    python aggregate_results.py --output work/results/summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


HEADLINE_METRICS = (
    "accuracy",
    "error_rate",
    "balanced_accuracy",
    "macro_auroc",
    "micro_auroc",
    "macro_auprc",
    "micro_auprc",
    "macro_f1",
    "micro_f1",
    "top1_hit_accuracy",
)


def load_row(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    metadata = payload.get("metadata", {})
    metrics = payload.get("metrics", {})

    row = {
        "file": path,
        "method": metadata.get("method"),
        "task": metadata.get("task"),
        "shots": metadata.get("shots"),
        "seed": metadata.get("seed"),
    }
    for metric_name in HEADLINE_METRICS:
        value = metrics.get(metric_name)
        row[metric_name] = np.nan if value is None else value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="work/results/*.json",
        help="glob pattern for result JSON files",
    )
    parser.add_argument(
        "--output",
        default="work/results/summary.csv",
        help="CSV path for the aggregated table",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No result files matched {args.pattern!r}")

    runs = pd.DataFrame(load_row(path) for path in paths)
    group_columns = ["method", "task", "shots"]
    metric_columns = [
        column
        for column in HEADLINE_METRICS
        if column in runs and runs[column].notna().any()
    ]

    grouped = runs.groupby(group_columns, dropna=False)[metric_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().add_suffix("_n")
    summary = pd.concat([mean, std, count], axis=1).reset_index()

    ordered_columns = list(group_columns)
    for metric_name in metric_columns:
        ordered_columns.extend(
            [f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_n"]
        )
    summary = summary[ordered_columns]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(summary.to_string(index=False))
    print(f"\nSaved aggregate table -> {output_path}")


if __name__ == "__main__":
    main()
