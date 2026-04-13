"""
sweep.py — Hyperparameter sweep for MSPlacer.

Patches config.toml with each parameter combination, runs the evaluator,
and prints a ranked results table.  Edit the SWEEP dict at the top to change
what gets swept.

Usage:
    # Single benchmark (fast):
    uv run python scripts/sweep.py

    # All benchmarks (slow but more reliable):
    uv run python scripts/sweep.py --all

    # Quiet — suppress per-run evaluator output:
    uv run python scripts/sweep.py --quiet

    # Keep the best config written to config.toml when done:
    uv run python scripts/sweep.py --keep-best
"""

# ── EDIT THIS SECTION ────────────────────────────────────────────────────────
# Each key must match a param name in config.toml [params].
# Each value is a list of candidate values to try.
# Every combination of values is evaluated (grid search).
SWEEP = {
    "target_density": [0.50, 0.60, 0.65, 0.70, 0.75, 0.80],
    # "lambda_pcof_upper": [1.03, 1.04, 1.05, 1.06],
    # "density_weight": [4e-5, 8e-5, 1.6e-4],
    # "warmup_iters": [10, 20, 40],
}
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

CONFIG_PATH = Path("submissions/msears/config.toml")
EVALUATOR   = "evaluate"
UV          = "/home/msears/.local/bin/uv"


def parse_args():
    p = argparse.ArgumentParser(description="Sweep MSPlacer hyperparameters.")
    p.add_argument("--all", action="store_true",
                   help="Run --all benchmarks instead of the default single benchmark")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-run evaluator output")
    p.add_argument("--keep-best", action="store_true",
                   help="Write the best-scoring config to config.toml when done")
    return p.parse_args()


def patch_config(original_text: str, params: dict) -> str:
    """Return config text with each param in `params` replaced."""
    text = original_text
    for key, val in params.items():
        # Match:  key  =  <any value>  (optional inline comment)
        pattern = rf"({re.escape(key)}\s*=\s*)([^\n#]+)"
        replacement = rf"\g<1>{val}"
        new_text, n = re.subn(pattern, replacement, text)
        if n == 0:
            print(f"  [warn] key '{key}' not found in config.toml — skipping", file=sys.stderr)
        text = new_text
    return text


def run_once(cmd: list, quiet: bool) -> tuple[float | None, str]:
    """Run evaluator, return (avg_proxy, raw_stdout)."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    stdout = r.stdout + r.stderr

    if not quiet:
        print(stdout, end="")

    # Single-benchmark result:  proxy=1.1234  (wl=... den=... cong=...) VALID [Xs]
    # Multi-benchmark AVG row:  AVG    1.6224  ...
    avg_match = re.search(r"AVG\s+([\d.]+)", stdout)
    if avg_match:
        return float(avg_match.group(1)), stdout

    proxy_matches = re.findall(r"proxy=([\d.]+)", stdout)
    if proxy_matches:
        return float(proxy_matches[-1]), stdout

    return None, stdout


def main():
    args = parse_args()

    original_config = CONFIG_PATH.read_text()

    # Build grid of all parameter combinations
    keys   = list(SWEEP.keys())
    values = list(SWEEP.values())
    combos = list(itertools.product(*values))
    n      = len(combos)

    cmd = [UV, "run", EVALUATOR, "submissions/msears/placer.py"]
    if args.all:
        cmd.append("--all")

    print(f"Sweep: {n} combination(s) × {'all benchmarks' if args.all else 'ibm01'}")
    print(f"Params: {keys}\n")

    results = []  # list of (proxy, params_dict, stdout)

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        label  = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i+1}/{n}]  {label}")

        patched = patch_config(original_config, params)
        CONFIG_PATH.write_text(patched)

        t0    = time.perf_counter()
        proxy, stdout = run_once(cmd, quiet=args.quiet)
        elapsed = time.perf_counter() - t0

        if proxy is not None:
            print(f"        proxy={proxy:.4f}  ({elapsed:.1f}s)\n")
            results.append((proxy, params, stdout))
        else:
            print(f"        [no result — check output above]\n")

    # Restore original config (or write best)
    if results and args.keep_best:
        best_proxy, best_params, _ = min(results, key=lambda r: r[0])
        best_config = patch_config(original_config, best_params)
        CONFIG_PATH.write_text(best_config)
        print(f"Wrote best config (proxy={best_proxy:.4f}) to config.toml")
    else:
        CONFIG_PATH.write_text(original_config)
        print("Restored original config.toml")

    if not results:
        print("No results collected.")
        return

    # Ranked table
    results.sort(key=lambda r: r[0])
    width = max(len("  ".join(f"{k}={v}" for k, v in p.items())) for _, p, _ in results)
    header = f"{'rank':<6}{'proxy':<10}  params"
    print(f"\n{'─' * (width + 20)}")
    print(header)
    print(f"{'─' * (width + 20)}")
    for rank, (proxy, params, _) in enumerate(results, 1):
        label = "  ".join(f"{k}={v}" for k, v in params.items())
        marker = "  ← best" if rank == 1 else ""
        print(f"{rank:<6}{proxy:<10.4f}  {label}{marker}")
    print(f"{'─' * (width + 20)}")
    print(f"Best: proxy={results[0][0]:.4f}  {results[0][1]}")


if __name__ == "__main__":
    main()
