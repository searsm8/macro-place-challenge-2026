#!/usr/bin/env python3
"""Analyze GA history across a sweep — tabulate per-gene statistics.

Reads every `ga_history.jsonl` under a sweep dir (one record per GA evaluation:
gen, ind, fitness, proxy, overlaps, genes, ...) and reports:

  - per-gene quartiles (full population vs top-K winners)
  - normalized location within each gene's GENE_SPACE range
  - choice-gene value counts (full pop vs winners)
  - Spearman correlation of each numeric gene with fitness
  - per-generation fitness trajectory (best/median/worst by gen)

Usage:
    uv run python scripts/analyze_ga_history.py sweep/sweep_20260518T022451Z
    uv run python scripts/analyze_ga_history.py sweep/<id> --top-k 0.25
    uv run python scripts/analyze_ga_history.py sweep/<id> --bench ibm01
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Import GENE_SPACE from the placer so ranges stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submissions.msears.ga_placer import GENE_SPACE


def load_history(sweep_dir: Path, bench_filter: str | None) -> pd.DataFrame:
    rows = []
    for fp in sweep_dir.rglob("ga_history.jsonl"):
        bench = fp.parent.name  # vis/frames/<bench>/ga_history.jsonl
        if bench_filter and bench != bench_filter:
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            row = {"benchmark": bench, **{k: r.get(k) for k in
                   ("gen", "ind", "run", "fitness", "proxy", "wl",
                    "density", "cong", "overlaps", "elapsed")}}
            row.update({f"g_{k}": v for k, v in r.get("genes", {}).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def normalize(val: float, spec: tuple) -> float:
    """Map a value to [0, 1] within its gene-space range."""
    kind = spec[0]
    if kind == "log":
        lo, hi = math.log(spec[1]), math.log(spec[2])
        return (math.log(val) - lo) / (hi - lo)
    lo, hi = spec[1], spec[2]
    return (val - lo) / (hi - lo)


def numeric_gene_table(df: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, spec in GENE_SPACE.items():
        if spec[0] not in ("linear", "log", "int"):
            continue
        col = f"g_{name}"
        if col not in df:
            continue
        full = df[col].dropna().astype(float)
        top  = winners[col].dropna().astype(float)
        q = full.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        qw = top.quantile([0.0, 0.25, 0.5, 0.75, 1.0]) if len(top) else pd.Series([np.nan]*5, index=q.index)
        try:
            nq = top.apply(lambda v: normalize(v, spec))
            n_q1, n_med, n_q3 = nq.quantile([0.25, 0.5, 0.75])
        except (ValueError, ZeroDivisionError):
            n_q1 = n_med = n_q3 = float("nan")
        out.append({
            "gene":   name,
            "kind":   spec[0],
            "range":  f"[{spec[1]:g}, {spec[2]:g}]",
            "n":      len(full),
            "min":    q[0.0],
            "q1":     q[0.25],
            "med":    q[0.5],
            "q3":     q[0.75],
            "max":    q[1.0],
            "top_q1": qw[0.25],
            "top_med": qw[0.5],
            "top_q3": qw[0.75],
            "norm_q1":  n_q1,
            "norm_med": n_med,
            "norm_q3":  n_q3,
        })
    return pd.DataFrame(out)


def choice_gene_table(df: pd.DataFrame, winners: pd.DataFrame) -> str:
    lines = []
    for name, spec in GENE_SPACE.items():
        if spec[0] != "choice":
            continue
        col = f"g_{name}"
        if col not in df:
            continue
        full_counts = Counter(df[col].dropna())
        top_counts  = Counter(winners[col].dropna())
        lines.append(f"\n  {name}  (choices: {spec[1]})")
        for v in spec[1]:
            f = full_counts.get(v, 0) + full_counts.get(str(v), 0)
            t = top_counts.get(v, 0) + top_counts.get(str(v), 0)
            lines.append(f"    {str(v):<12}  full={f:4d}   top={t:4d}")
    return "\n".join(lines)


def spearman_with_fitness(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["fitness"].notna()].copy()
    out = []
    for name, spec in GENE_SPACE.items():
        if spec[0] not in ("linear", "log", "int"):
            continue
        col = f"g_{name}"
        if col not in sub:
            continue
        s = sub[[col, "fitness"]].dropna()
        if len(s) < 5:
            continue
        rho = s[col].rank().corr(s["fitness"].rank())
        out.append({"gene": name, "spearman_rho_vs_fitness": rho, "n": len(s)})
    return pd.DataFrame(out).sort_values("spearman_rho_vs_fitness", key=lambda s: s.abs(), ascending=False)


def gen_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["fitness"].notna()].groupby(["benchmark", "gen"])["fitness"]
    return g.agg(["min", "median", "max", "count"]).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--bench", default=None, help="Filter to a single benchmark")
    ap.add_argument("--top-k", type=float, default=0.25,
                    help="Fraction (0-1) or count (>1) of best-fitness rows to treat as 'winners'")
    ap.add_argument("--csv", type=Path, default=None, help="Write numeric table to CSV")
    args = ap.parse_args()

    df = load_history(args.sweep_dir, args.bench)
    if df.empty:
        sys.exit(f"No ga_history.jsonl found under {args.sweep_dir}")

    # winners = top-K by fitness (within-benchmark percentile so all benches contribute)
    df_valid = df[df["fitness"].notna()].copy()
    if args.top_k <= 1.0:
        df_valid["pct"] = df_valid.groupby("benchmark")["fitness"].rank(pct=True)
        winners = df_valid[df_valid["pct"] <= args.top_k]
    else:
        k = int(args.top_k)
        winners = df_valid.groupby("benchmark", group_keys=False).apply(
            lambda g: g.nsmallest(k, "fitness"))

    print(f"\nLoaded {len(df)} records across {df['benchmark'].nunique()} benchmark(s)")
    print(f"Valid (finite fitness): {len(df_valid)}   winners (top {args.top_k}): {len(winners)}\n")

    print("=" * 110)
    print("NUMERIC GENES  —  full-population quartiles + winner quartiles + winner location in [0,1]")
    print("=" * 110)
    tbl = numeric_gene_table(df, winners)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(tbl.to_string(index=False))

    print("\n" + "=" * 110)
    print("CHOICE GENES  —  full-population vs winner counts")
    print("=" * 110)
    print(choice_gene_table(df, winners))

    print("\n" + "=" * 110)
    print("SPEARMAN CORRELATION  —  gene rank vs fitness rank (negative = lower gene → better fitness)")
    print("=" * 110)
    corr = spearman_with_fitness(df)
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(corr.to_string(index=False))

    print("\n" + "=" * 110)
    print("PER-GENERATION FITNESS  —  is the GA actually improving?")
    print("=" * 110)
    traj = gen_trajectory(df)
    print(traj.to_string())

    if args.csv:
        tbl.to_csv(args.csv, index=False)
        print(f"\nNumeric table written to {args.csv}")


if __name__ == "__main__":
    main()
