#!/usr/bin/env python3
"""Plot sweep results from a CSV, grouped by criteria columns.

Each criterion value gets its own figure with 4 subplots (proxy, wl, den, cong).
X-axis points represent parameter combinations; labels show the varying sweep params.

Usage:
    uv run python scripts/analyze_results.py results.csv --criteria benchmark mip_only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["proxy", "wl", "den", "cong"]
SKIP_COLS = {"combo_id", "valid", "time_s", "mgp_iters", "cgp_iters", "overlaps", "sweep_id"}


def resolve_col(name: str, columns: list[str]) -> str | None:
    """Allow 'mip_only' to match 'sweep_mip_only'."""
    if name in columns:
        return name
    candidate = f"sweep_{name}"
    if candidate in columns:
        return candidate
    return None


def label_params_for(df: pd.DataFrame, exclude_col: str) -> list[str]:
    """Columns that vary in df and are not metrics/metadata/the split criterion."""
    skip = set(METRICS) | SKIP_COLS | {exclude_col}
    return [c for c in df.columns if c not in skip and df[c].nunique() > 1]


def row_label(row: pd.Series, params: list[str]) -> str:
    parts = []
    for p in params:
        short = p.removeprefix("sweep_")
        val = row[p]
        if isinstance(val, float):
            val = f"{val:.3g}"
        parts.append(f"{short}={val}")
    return "\n".join(parts)


def plot_group(df: pd.DataFrame, title: str, label_params: list[str], out_path: Path):
    sort_cols = label_params if label_params else [df.columns[0]]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    labels = [row_label(row, label_params) for _, row in df.iterrows()]
    x = np.arange(len(df))
    is_valid = df["valid"].str.upper() == "VALID" if "valid" in df.columns else pd.Series([True] * len(df))

    fig_w = max(10, len(df) * 0.7 + 2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flatten(), METRICS):
        y = df[metric].values

        # Plot valid and invalid points differently
        valid_mask = is_valid.values
        ax.plot(x[valid_mask], y[valid_mask], "o-", markersize=5, linewidth=1.3, label="valid")
        if (~valid_mask).any():
            ax.plot(x[~valid_mask], y[~valid_mask], "rx", markersize=7, label="invalid")

        # Highlight best (min) among valid
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices):
            best_i = valid_indices[int(np.argmin(y[valid_indices]))]
            ax.plot(x[best_i], y[best_i], "*", color="gold", markersize=12,
                    zorder=5, label=f"best={y[best_i]:.4f}")

        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel(metric)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Path to results CSV")
    parser.add_argument("--criteria", nargs="+", default=["benchmark"],
                        help="Columns to split plots by (e.g. benchmark mip_only)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <csv_dir>/plots/)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = []
    for name in args.criteria:
        col = resolve_col(name, list(df.columns))
        if col is None:
            print(f"Warning: '{name}' not found in columns, skipping", file=sys.stderr)
        else:
            criteria.append(col)

    if not criteria:
        sys.exit("Error: no valid criteria columns found")

    n_plots = sum(df[c].nunique() for c in criteria)
    print(f"Criteria: {criteria} → {n_plots} plots → {out_dir}/\n")

    for criterion in criteria:
        for value in sorted(df[criterion].unique(), key=str):
            subset = df[df[criterion] == value].copy()
            label_params = label_params_for(subset, exclude_col=criterion)
            safe_val = str(value).replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"{criterion.removeprefix('sweep_')}_{safe_val}.png"
            title = f"{criterion.removeprefix('sweep_')} = {value}  (n={len(subset)})"
            print(f"{title}  labels={[p.removeprefix('sweep_') for p in label_params]}")
            plot_group(subset, title, label_params, out_path)

    print(f"\nDone. {n_plots} plots in {out_dir}/")


if __name__ == "__main__":
    main()
