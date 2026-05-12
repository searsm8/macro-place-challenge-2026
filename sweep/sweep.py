"""
sweep/sweep.py — Parallel hyperparameter sweep for CometPlacer.

Each (config combo × benchmark) is an independent job dispatched to a thread
pool. Results are saved per-run and collated into sweep/results.csv.

Directory layout
----------------
sweep/
  sweep.py                  ← this script
  results.csv               ← master CSV (all sweeps appended here)
  sweep_<timestamp>/        ← one directory per sweep invocation
    results.csv             ← per-sweep CSV
    <benchmark>/
      run_001/
        config.toml         ← the exact config used for this job
        run_summary.md      ← human-readable result summary

Usage
-----
    # cd to repo root first, then:
    uv run python sweep/sweep.py                         # ibm01, 8 parallel workers
    uv run python sweep/sweep.py --all --workers 16      # all benchmarks, 16 workers
    uv run python sweep/sweep.py -b ibm03 --quiet        # specific benchmark
    uv run python sweep/sweep.py --keep-best             # write best to config.toml
"""

# ── EDIT THIS SECTION ────────────────────────────────────────────────────────
# SWEEP: keys are config.toml param names; values are lists of candidates.
#   The full Cartesian product of SWEEP is evaluated.
#   If COMBOS is non-empty, every SWEEP grid point is crossed with every COMBO.
#
# COMBOS: explicit parameter groups (list of dicts).
#   Each dict is merged on top of the SWEEP grid point for that run.
#   Use this to test hand-picked combinations without a full Cartesian product.
#   Example:
#     SWEEP  = {"seed": [0, 1, 2, 3]}
#     COMBOS = [
#         {"mgp_enable": "true",  "legalization": "bump"},
#         {"mgp_enable": "false", "legalization": "none"},
#     ]
#   → 4 seeds × 2 combos = 8 runs per benchmark.
#
# Leave COMBOS = [] to use plain Cartesian product (original behaviour).
COMBOS: list[dict] = [
     #{"mgp_enable": "true",  "legalization": "bump"},
     #{"mgp_enable": "false", "legalization": "none"},
     #{"initial_placement": "none"},
     #{"initial_placement": "center"},

     #{"rotation_optimizer": "none"},
     #{"rotation_optimizer": "greedy", "n_placement_passes": 2},
    #{"cGP_enable": "False"},
]

SWEEP = {
    #"initial_spread": [0.01, 0.04, 0.15],
    #"warmup_iters": [0, 1, 10, 20, 40],
    #"optimizer": ["sgd", "bb_sgd", "nesterov"],

    #"lambda_pcof_upper": [1.03, 1.05],
    #"lambda_hm_init": [1, 2, 3, 5, 10, 20], #(4/23) 1 is best (disabled density boost)
    #"soft_place_iters": [1000, 3000, 5000],
    #"lambda_max": [25000],
    #"initial_placement": ["none", "center", "quadratic"],
    #"initial_placement": ["none", "quadratic"],
    #"initial_placement": ["center"],
    #"center_init_spread": [0.15],
    #"density_weight": [1e-3],
    #"lambda_iters_per_update": [1, 3, 5],
    #"curtailed_iters": [300, 500, 700, 900]

    #"stop_overflow": [0.03, 0.04, .05],
    #"rotation_optimizer": ["none", "greedy"],
    #"use_gpu": ["False", "True"],
    #"deterministic": ["False","False","True","True"],
    #"n_placement_passes": [1, 2, 3],
    #"rotation_candidates": ["all", "no-swap"],
    #"gamma_decay": [0.991, 0.992, 0.993, 0.994],
    #"gamma": ["auto", "ovfw"],

    #"quad_b2b_iters": [0, 1, 3, 6],     
    #"quad_net_size_threshold": [20, 80, 150],
    #"quad_anchor_fraction": [ 0, .1, .2, 0.97], # fraction of movable macros (ranked by area×net_degree) fixed as internal anchors in stage 1 (0 = disabled)
    #"density_grid_size": [75, 128, 256, 512],
    #"target_density": [0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90],
    #"density_grid_size": [64, 96, 128, 192],
    #"target_density": [0.65],
    #"halo_legalize": [0.1, 0.15],
    #"quad_scatter_fraction": [0, 0.1, 0.2, 0.3], # fraction of movable macros (ranked by area×net_degree) randomly scattered in stage 1 (0 = disabled)
    #"quad_scatter_lock_mult": [5, 10, 20, 50], # keep scatter macros fixed in mGP until lambda >= this × lambda_0
    #"mip_only": ["False", "True"], # skip mGP/cGP — legalize from mIP positions and return immediately (fast proxy for scatter quality)

    #"lambda_cong": [0.32], # congestion penalty weight (relative to density) in mGP
    #"lambda_cong_step": [ 1.01], # multiplicative increase of lambda_cong per mGP iteration (if >1, otherwise fixed)
    #"quad_scatter_fraction": [0, 0.1, 0.15, 0.2], # keep scatter macros fixed in mGP until lambda >= this × lambda_0
    #"quad_scatter_n": [0, 1, 2, 4, 8, 12, 15, 20],  # keep scatter macros fixed in mGP until lambda >= this × lambda_0
    #"quad_scatter_lock_mult": [0, 1e3, 1e4, 1e6], # keep scatter macros fixed in mGP until lambda >= this × lambda_0
    #"n_placement_passes": [1, 2]

    #"cGP_enable": ["True"],
    #"cGP_lambda_cong_init": [1e-4, 1e-3, 5e-3, 1e-2, 1.5e-2],
    #"seed": [999, 9999, 99999, 999999, 888, 777, 666, 555], # random seed for initial placement and mGP; affects all stochasticity when deterministic=false
    #"halo_size": [ 0.2, 0.25, 0.3, 0.35],
    #"lambda_density_init": [1e-7, 1e-6, 1e-5],
    #"cong_rudy_grid_size": [32, 64, 128],
    #"max_step": [0.002, 0.0025, 0.003, 0.0035, 0.004],
    #"hard_spread": ["False", "True"],
    #"hard_spread_iters": [50, 100, 400],
    "cGP_hard_density ": ["False", "True"],
}
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
SUBMISSION_DIR = REPO_ROOT / "submissions/msears"
CONFIG_PATH = REPO_ROOT / SUBMISSION_DIR / "config.toml"
SWEEP_DIR   = Path(__file__).parent          # …/sweep/
MASTER_CSV  = SWEEP_DIR / "results.csv"
UV          = "/home/msears/.local/bin/uv"

IBM_BENCHMARKS = [

    # Full set
    #"ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    #"ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",

    # Small subset
    "ibm01", "ibm04", "ibm14", # best avg for this subset: 1.3782 (new best, 1.2525)
    
    # Medium Subset
    #"ibm01", "ibm02",  "ibm04", "ibm07", "ibm11", "ibm14",
]

METRIC_COLS = ["proxy", "wl", "den", "cong", "valid", "time_s", "mgp_iters", "cgp_iters", "overlaps"]


# ── Config helpers ─────────────────────────────────────────────────────────────

def toml_value(val) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def patch_config(original_text: str, params: dict) -> str:
    """Return config text with each param in *params* overwritten.

    Matches the value token (quoted string OR bare word/number), leaving any
    trailing whitespace and inline comment untouched.
    """
    text = original_text
    for key, val in params.items():
        pattern  = rf'({re.escape(key)}\s*=\s*)("[^"]*"|[^\s\n#]+)'
        tval     = toml_value(val)
        new_text, n = re.subn(pattern, rf"\g<1>{tval}", text)
        if n == 0:
            print(
                f"  [warn] key '{key}' not found in config.toml — skipping",
                file=sys.stderr,
            )
        text = new_text
    return text


# ── Output parsing ─────────────────────────────────────────────────────────────

def parse_result(stdout: str, benchmark: str) -> dict | None:
    """Parse evaluator stdout for a single benchmark run.

    Returns a dict with METRIC_COLS keys, or None if no result was found.
    The evaluator prints one result line per benchmark:
        ibm01... proxy=1.0594  (wl=0.063 den=0.761 cong=1.220)  VALID  [2.34s]
        ibm01... proxy=1.0594  (wl=...) INVALID (142 overlaps)  [2.34s]
    """
    result: dict = {
        "benchmark": benchmark,
        "proxy":     None,
        "wl":        None,
        "den":       None,
        "cong":      None,
        "valid":     None,
        "time_s":    None,
        "mgp_iters": None,
        "cgp_iters": None,
        "overlaps":  None,
    }

    for line in stdout.splitlines():
        # Main result line
        m = re.search(
            r"proxy=([\d.]+)\s+\(wl=([\d.]+)\s+den=([\d.]+)\s+cong=([\d.]+)\)"
            r"\s+(VALID|INVALID(?:\s+\(\d+\s+overlaps\))?)\s+\[([\d.]+)s\]",
            line,
        )
        if m:
            result["proxy"]  = float(m.group(1))
            result["wl"]     = float(m.group(2))
            result["den"]    = float(m.group(3))
            result["cong"]   = float(m.group(4))
            result["valid"]  = "VALID" if m.group(5) == "VALID" else "INVALID"
            result["time_s"] = float(m.group(6))

        # Inline overlap count: "INVALID (142 overlaps)"
        m2 = re.search(r"INVALID\s+\((\d+)\s+overlaps\)", line)
        if m2:
            result["overlaps"] = int(m2.group(1))

        # Iteration counts: "[mGP] done  iters=N" and "[cGP] done  iters=N"
        m3 = re.search(r"\[mGP\] done\s+iters=(\d+)", line)
        if m3:
            result["mgp_iters"] = int(m3.group(1))
        m4 = re.search(r"\[cGP\] done\s+iters=(\d+)", line)
        if m4:
            result["cgp_iters"] = int(m4.group(1))

    if result["proxy"] is None:
        return None

    if result["valid"] == "VALID" and result["overlaps"] is None:
        result["overlaps"] = 0

    return result


# ── Run summary ────────────────────────────────────────────────────────────────

def write_run_summary(
    path: Path,
    benchmark: str,
    combo_id: int,
    sweep_params: dict,
    result: dict | None,
):
    lines = [
        f"# Run {combo_id:03d} — {benchmark}\n\n",
        f"**Sweep params:** `{sweep_params}`\n\n",
    ]

    if result and result["proxy"] is not None:
        lines += [
            "## Results\n\n",
            "| Metric   | Value |\n",
            "|----------|-------|\n",
            f"| proxy    | {result['proxy']:.4f} |\n",
            f"| wl       | {result['wl']:.4f} |\n",
            f"| den      | {result['den']:.4f} |\n",
            f"| cong     | {result['cong']:.4f} |\n",
            f"| valid    | {result['valid']} |\n",
            f"| overlaps | {result['overlaps']} |\n",
            f"| time_s   | {result['time_s']:.2f} |\n",
        ]
    else:
        lines.append("## Results\n\n**FAILED — no output parsed.**\n")

    path.write_text("".join(lines))


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _sweep_csv_columns(sweep_keys: list) -> list:
    """Columns for the per-sweep CSV — includes the specific sweep params."""
    return (
        ["combo_id", "benchmark"]
        + METRIC_COLS
        + [f"sweep_{k}" for k in sweep_keys]
        + ["sweep_id"]
    )


MASTER_CSV_COLUMNS = ["combo_id", "benchmark"] + METRIC_COLS + ["sweep_id"]


def _build_csv_row(
    sweep_id: str,
    combo_id: int,
    sweep_params: dict,
    result: dict,
) -> dict:
    row = {
        "combo_id":  combo_id,
        "benchmark": result.get("benchmark", ""),
        "sweep_id":  sweep_id,
    }
    row.update({f"sweep_{k}": v for k, v in sweep_params.items()})
    for col in METRIC_COLS:
        row[col] = result.get(col, "")
    return row


def write_csv(path: Path, rows: list[dict], columns: list, append: bool = False):
    mode   = "a" if append else "w"
    is_new = append and (not path.exists() or path.stat().st_size == 0)
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not append or is_new:
            writer.writeheader()
        writer.writerows(rows)


# ── Worker ─────────────────────────────────────────────────────────────────────

def run_one(
    run_dir: Path,
    benchmark: str,
    combo_id: int,
    sweep_params: dict,
    original_config: str,
    quiet: bool,
    record_frames: bool,
    render_gifs: bool,
) -> dict | None:
    """Execute one (config combo × benchmark) job.

    1. Writes patched config to run_dir/config.toml
    2. Invokes the evaluator with MSPLACER_CONFIG pointing at that file
       - If record_frames: frames saved to run_dir/frames/
       - If render_gifs:   GIF rendered to run_dir/placement.gif after the run
    3. Parses stdout, writes run_summary.md
    4. Returns a result dict (or None on failure)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = run_dir / "config.toml"

    patched = patch_config(original_config, sweep_params)
    config_file.write_text(patched)

    env = os.environ.copy()
    env["MSPLACER_CONFIG"] = str(config_file.resolve())
    if record_frames:
        env["MSPLACER_FRAMES_DIR"] = str((run_dir / "frames").resolve())
    else:
        env["MSPLACER_NO_FRAMES"] = "1"

    cmd = [
        UV, "run", "evaluate",
        str(REPO_ROOT / SUBMISSION_DIR / "placer.py"),
        "-b", benchmark,
    ]

    t0    = time.perf_counter()
    proc  = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(REPO_ROOT), env=env,
    )
    stdout  = proc.stdout + proc.stderr
    elapsed = time.perf_counter() - t0

    if not quiet:
        print(stdout, end="")

    result = parse_result(stdout, benchmark)

    write_run_summary(
        run_dir / "run_summary.md",
        benchmark, combo_id, sweep_params, result,
    )

    label = "  ".join(f"{k}={v}" for k, v in sweep_params.items())
    if result:
        status = (
            f"proxy={result['proxy']:.4f}  "
            f"overlaps={result['overlaps']}  ({elapsed:.1f}s)"
        )
    else:
        status = f"[FAILED]  ({elapsed:.1f}s)"
    print(f"  [{combo_id:03d}] {benchmark:<8}  {label}  →  {status}")

    # Render GIF if frames were recorded
    if record_frames and render_gifs:
        gif_path = run_dir / "placement.gif"
        gif_cmd  = [
            UV, "run", "python",
            str(REPO_ROOT / "scripts/frames_to_gif.py"),
            "--benchmark",  benchmark,
            "--frames-dir", str((run_dir / "frames").resolve()),
            "--output",     str(gif_path.resolve()),
            "--step",       "10", # use every 10th frame to speed up rendering
        ]
        subprocess.run(gif_cmd, capture_output=True, cwd=str(REPO_ROOT))
        if gif_path.exists():
            print(f"  [{combo_id:03d}] {benchmark:<8}  GIF → {gif_path.relative_to(REPO_ROOT)}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel MSPlacer hyperparameter sweep.",
    )
    p.add_argument("--all", action="store_true",
                   help="Run all 17 IBM benchmarks (default: ibm01)")
    p.add_argument("--benchmark", "-b", default=None,
                   help="Run a specific benchmark (e.g. ibm03)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-run evaluator stdout")
    p.add_argument("--no-record-frames", dest="record_frames", action="store_false", default=True,
                   help="Skip frame snapshots (faster; disables --render-gifs)")
    p.add_argument("--render-gifs", action="store_true",
                   help="Render a placement.gif for each run after it completes")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel workers (default: 8)")
    p.add_argument("--keep-best", action="store_true",
                   help="Write the best config to submissions/msears/config.toml")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    original_config = CONFIG_PATH.read_text()

    # Build list of parameter dicts: SWEEP grid crossed with COMBOS (if any).
    _sweep_vals = list(SWEEP.values()) if SWEEP else [()]
    _sweep_dicts = [
        dict(zip(SWEEP.keys(), c)) for c in itertools.product(*_sweep_vals)
    ] if SWEEP else [{}]
    if COMBOS:
        param_dicts = [{**sd, **co} for sd in _sweep_dicts for co in COMBOS]
    else:
        param_dicts = _sweep_dicts
    sweep_keys = list(dict.fromkeys(k for d in param_dicts for k in d))
    n_combos   = len(param_dicts)

    if args.all:
        benchmarks = IBM_BENCHMARKS
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = ["ibm01"]

    # Create per-sweep output directory
    ts            = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_run_dir = SWEEP_DIR / f"sweep_{ts}"
    sweep_run_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = n_combos * len(benchmarks)
    print(f"Sweep : {n_combos} combo(s) × {len(benchmarks)} benchmark(s) = {n_jobs} jobs")
    print(f"Output : {sweep_run_dir.relative_to(REPO_ROOT)}")
    print(f"Master : {MASTER_CSV.relative_to(REPO_ROOT)}")
    print()

    # Build job list: benchmark-outer so all combos for one benchmark run before
    # moving to the next.
    jobs: dict[str, list] = {b: [] for b in benchmarks}
    for benchmark in benchmarks:
        for combo_id, params in enumerate(param_dicts, 1):
            run_dir = sweep_run_dir / benchmark / f"run_{combo_id:03d}"
            jobs[benchmark].append((run_dir, benchmark, combo_id, params))

    # ── Execute in parallel, one benchmark at a time ──────────────────────────
    collected: list[tuple[int, dict, dict]] = []   # (combo_id, params, result)

    record_frames = args.record_frames
    render_gifs   = args.render_gifs
    n_workers     = min(args.workers, n_jobs) if n_jobs > 0 else 1

    # Initialise both CSVs with headers before the loop so partial results are
    # written immediately after each run (safe to interrupt mid-sweep).
    sweep_csv      = sweep_run_dir / "results.csv"
    sweep_csv_cols = _sweep_csv_columns(sweep_keys)
    write_csv(sweep_csv,  [], sweep_csv_cols,      append=False)
    write_csv(MASTER_CSV, [], MASTER_CSV_COLUMNS,  append=True)

    csv_lock = threading.Lock()
    print(f"Workers: {n_workers}")

    futures = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for benchmark in benchmarks:
            for run_dir, bm, combo_id, params in jobs[benchmark]:
                fut = executor.submit(
                    run_one,
                    run_dir, bm, combo_id, params,
                    original_config, args.quiet,
                    record_frames, render_gifs,
                )
                futures[fut] = (combo_id, params)

        for fut in as_completed(futures):
            combo_id, params = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  [ERROR] combo={combo_id:03d}: {exc}")
                result = None
            if result:
                collected.append((combo_id, params, result))
                row = _build_csv_row(ts, combo_id, params, result)
                with csv_lock:
                    write_csv(sweep_csv,  [row], sweep_csv_cols,     append=True)
                    write_csv(MASTER_CSV, [row], MASTER_CSV_COLUMNS, append=True)

    # ── CSV paths already written incrementally; just print locations ─────────
    print(f"\nSweep CSV : {sweep_csv.relative_to(REPO_ROOT)}")
    print(f"Master CSV: {MASTER_CSV.relative_to(REPO_ROOT)}")

    # ── Ranked summary ────────────────────────────────────────────────────────
    if not collected:
        print("\nNo results collected.")
        return

    combo_proxies: dict[int, list[float]] = defaultdict(list)
    combo_params_map: dict[int, dict]     = {}
    for combo_id, params, result in collected:
        if result.get("proxy") is not None:
            combo_proxies[combo_id].append(result["proxy"])
            combo_params_map[combo_id] = params

    ranked = sorted(
        [
            (cid, sum(ps) / len(ps), len(ps), combo_params_map[cid])
            for cid, ps in combo_proxies.items()
        ],
        key=lambda x: x[1],
    )

    if not ranked:
        return

    width  = max(
        len("  ".join(f"{k}={v}" for k, v in p.items())) for *_, p in ranked
    )
    sep    = "─" * (width + 34)
    header = f"{'rank':<6}{'avg_proxy':<12}{'n_bench':<9}  params"
    print(f"\n{sep}\n{header}\n{sep}")
    for rank, (cid, ap, nb, params) in enumerate(ranked, 1):
        label  = "  ".join(f"{k}={v}" for k, v in params.items())
        marker = "  ← best" if rank == 1 else ""
        print(f"{rank:<6}{ap:<12.4f}{nb:<9}  {label}{marker}")
    print(sep)

    best_cid, best_ap, best_nb, best_params = ranked[0]
    print(f"Best: avg_proxy={best_ap:.4f}  {best_params}")

    if args.keep_best:
        CONFIG_PATH.write_text(patch_config(original_config, best_params))
        print(f"Wrote best config to {CONFIG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
