#!/usr/bin/env python3
"""Plot benchmark results from CSV.

Usage:
    # Run the benchmark first:
    cd crates/bloqade-lanes-search
    cargo test -p bloqade-lanes-search benchmark_sweep_random -- --nocapture --ignored

    # Then plot:
    python scripts/plot_benchmark.py benchmark_results.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute mean/std across seeds per (qubits, approach).
    grouped = (
        df.groupby(["qubits", "approach"])
        .agg(
            cost_mean=("total_cost", "mean"),
            cost_std=("total_cost", "std"),
            expanded_mean=("total_expanded", "mean"),
            expanded_std=("total_expanded", "std"),
            time_mean=("time_ms", "mean"),
            time_std=("time_ms", "std"),
            failures_mean=("failures", "mean"),
            cost_per_layer_mean=("cost_per_layer", "mean"),
            cost_per_layer_std=("cost_per_layer", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return df, grouped


APPROACH_STYLES = {
    "baseline_palindrome": {"color": "#d62728", "marker": "s", "label": "Baseline (palindrome)"},
    "loose_static": {"color": "#2ca02c", "marker": "o", "label": "Loose static"},
    "loose_dyn_deadlock": {"color": "#e377c2", "marker": "*", "label": "Dynamic (deadlock)"},
    "loose_dyn_1": {"color": "#1f77b4", "marker": "D", "label": "Dynamic (every exp)"},
    "loose_dyn_5": {"color": "#9467bd", "marker": "^", "label": "Dynamic (interval=5)"},
    "loose_dyn_10": {"color": "#ff7f0e", "marker": "v", "label": "Dynamic (interval=10)"},
    "loose_dyn_50": {"color": "#8c564b", "marker": "p", "label": "Dynamic (interval=50)"},
}


def plot_metric(
    ax, grouped, metric_mean, metric_std, ylabel, title, log_y=False
):
    for approach, style in APPROACH_STYLES.items():
        data = grouped[grouped["approach"] == approach].sort_values("qubits")
        if data.empty:
            continue
        x = data["qubits"].values
        y = data[metric_mean].values
        yerr = data[metric_std].values if metric_std else None

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.5,
            capsize=3,
        )

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("csv", help="Path to benchmark_results.csv")
    parser.add_argument(
        "-o",
        "--output",
        default="benchmark_results.png",
        help="Output image path (default: benchmark_results.png)",
    )
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"Error: {args.csv} not found.", file=sys.stderr)
        print(
            "Run the benchmark first:\n"
            "  cargo test -p bloqade-lanes-search benchmark_sweep_random "
            "-- --nocapture --ignored",
            file=sys.stderr,
        )
        sys.exit(1)

    df, grouped = load_data(args.csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Loose-Goal Search vs Baseline: Random CZ Circuits (depth=10, 5 seeds)",
        fontsize=14,
        fontweight="bold",
    )

    plot_metric(
        axes[0, 0],
        grouped,
        "cost_per_layer_mean",
        "cost_per_layer_std",
        "Cost per layer (move steps)",
        "Cost per CZ Layer",
    )

    plot_metric(
        axes[0, 1],
        grouped,
        "total_cost_mean" if "total_cost_mean" in grouped.columns else "cost_mean",
        "cost_std",
        "Total cost (move steps)",
        "Total Cost (10 layers)",
    )

    plot_metric(
        axes[1, 0],
        grouped,
        "time_mean",
        "time_std",
        "Wall-clock time (ms)",
        "Solve Time",
        log_y=True,
    )

    plot_metric(
        axes[1, 1],
        grouped,
        "expanded_mean",
        "expanded_std",
        "Nodes expanded",
        "Search Effort",
        log_y=True,
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

    # Also print summary table.
    print("\nSummary (mean across seeds):")
    print(
        grouped[
            [
                "qubits",
                "approach",
                "cost_per_layer_mean",
                "time_mean",
                "expanded_mean",
                "failures_mean",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
