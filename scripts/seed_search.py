"""
Run the same benchmark with many different seeds and report the best result.

Usage:
    uv run python scripts/seed_search.py
    uv run python scripts/seed_search.py --benchmark ibm01 --seeds 0 1 2 3 4
    uv run python scripts/seed_search.py --seeds $(seq 0 19)
"""

import argparse
import copy
import sys
import time
import tomllib
from pathlib import Path

# Make sure the repo root is on the path so imports work.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


def load_base_config():
    config_path = ROOT / "submissions/msears/config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def run_seed(seed: int, benchmark_dir: str, base_config: dict) -> dict:
    # Reload benchmark fresh each run — placer mutates benchmark.macro_sizes
    # in-place during rotation (w↔h swap for E/W orientations).
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    cfg = copy.deepcopy(base_config)
    cfg.setdefault("params", {})["seed"] = seed
    # Disable frame recording to keep runs fast and avoid disk I/O.
    cfg.setdefault("output", {})["record_frames"] = False
    cfg["output"]["quiet"] = True

    from submissions.msears.placer import CometPlacer
    placer = CometPlacer(config=cfg)

    t0 = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - t0

    _, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)

    return {
        "seed": seed,
        "proxy": costs["proxy_cost"],
        "wl": costs["wirelength_cost"],
        "density": costs["density_cost"],
        "congestion": costs["congestion_cost"],
        "overlaps": costs["overlap_count"],
        "runtime": runtime,
    }


def main():
    parser = argparse.ArgumentParser(description="Seed sweep for CometPlacer.")
    parser.add_argument("--benchmark", "-b", default="ibm01")
    parser.add_argument("--seeds", "-s", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()

    benchmark_dir = str(ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}")
    base_config = load_base_config()

    print(f"Benchmark: {args.benchmark}  |  Seeds: {args.seeds}")
    print(f"{'Seed':>6}  {'Proxy':>8}  {'WL':>8}  {'Density':>8}  {'Congestion':>10}  {'Overlaps':>8}  {'Time(s)':>8}")
    print("-" * 72)

    results = []
    for seed in args.seeds:
        r = run_seed(seed, benchmark_dir, base_config)
        results.append(r)
        flag = "  ← best" if r["proxy"] == min(x["proxy"] for x in results) else ""
        print(
            f"{r['seed']:>6}  {r['proxy']:>8.4f}  {r['wl']:>8.3f}  {r['density']:>8.3f}"
            f"  {r['congestion']:>10.3f}  {r['overlaps']:>8}  {r['runtime']:>8.1f}{flag}"
        )

    best = min(results, key=lambda r: r["proxy"])
    print("-" * 72)
    print(f"\nBest seed: {best['seed']}  proxy={best['proxy']:.4f}  overlaps={best['overlaps']}")
    print(f"\nTo use: set seed = {best['seed']} in submissions/msears/config.toml")


if __name__ == "__main__":
    main()
