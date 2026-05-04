"""
sweep_to_gifs.py — Render GIFs for every run in a completed sweep directory.

After running a sweep with frame recording but without --render-gifs, use this
script to generate all the GIFs in one go.

Directory structure expected:
    sweep/sweep_<timestamp>/
        <benchmark>/
            run_001/
                frames/<benchmark>/frame_*.pt
                frames/<benchmark>/net_edges.pt
            run_002/
                frames/<benchmark>/frame_*.pt
            ...

A placement.gif is written inside each run directory.

Usage:
    # Latest sweep, all runs
    uv run python scripts/sweep_to_gifs.py

    # Specific sweep directory
    uv run python scripts/sweep_to_gifs.py sweep/sweep_20260421T120000Z

    # Only runs for one benchmark
    uv run python scripts/sweep_to_gifs.py --benchmark ibm03

    # Faster rendering: every 10th frame, lower DPI
    uv run python scripts/sweep_to_gifs.py --step 10 --dpi 60 --fps 20

    # Skip runs that already have a placement.gif
    uv run python scripts/sweep_to_gifs.py --skip-existing
"""

import argparse
import importlib.util
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "sweep"

# Load frames_to_gif helpers once so all runs share the same module
_ftg_path = REPO_ROOT / "scripts" / "frames_to_gif.py"
_spec = importlib.util.spec_from_file_location("frames_to_gif", _ftg_path)
ftg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ftg)


def find_latest_sweep() -> Path | None:
    dirs = sorted(SWEEP_DIR.glob("sweep_*"), reverse=True)
    return dirs[0] if dirs else None


def _is_benchmark_dir(path: Path) -> bool:
    """True if path looks like sweep_XXX/benchmark/ (contains run_NNN children)."""
    return any(d.is_dir() and d.name.startswith("run_") for d in path.iterdir())


def find_runs(sweep_path: Path, benchmark_filter: set[str] | None) -> list[tuple[Path, str]]:
    """Return (run_dir, benchmark_name) pairs whose frames/ subdir has frames.

    Accepts either a sweep root (sweep_XXX/) or a single benchmark directory
    (sweep_XXX/ibm01/) — the latter lets you render one benchmark without
    needing the --benchmark flag.
    """
    # Detect if the user passed a benchmark-level dir directly
    if _is_benchmark_dir(sweep_path):
        bench_dirs = [(sweep_path, sweep_path.name)]
    else:
        bench_dirs = [(d, d.name) for d in sorted(sweep_path.iterdir()) if d.is_dir()]

    runs = []
    for bench_dir, bench_name in bench_dirs:
        if benchmark_filter and bench_name not in benchmark_filter:
            continue
        for run_dir in sorted(bench_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            frames_bench = run_dir / "frames" / bench_name
            if frames_bench.is_dir() and any(frames_bench.glob("frame_*.pt")):
                runs.append((run_dir, bench_name))
    return runs


def parse_args():
    p = argparse.ArgumentParser(
        description="Render GIFs for every run in a sweep directory."
    )
    p.add_argument("sweep_dir", nargs="?", default=None,
                   help="Path to sweep_<timestamp> directory "
                        "(default: latest sweep in sweep/)")
    p.add_argument("--benchmark", "-b", nargs="+", default=None, metavar="BENCH",
                   help="Only render runs for these benchmark(s) "
                        "(e.g. -b ibm01 ibm03 ibm06)")
    p.add_argument("--fps", type=float, default=10,
                   help="Frames per second (default: 10)")
    p.add_argument("--step", type=int, default=10,
                   help="Render every Nth frame (default: 10)")
    p.add_argument("--dpi", type=int, default=80,
                   help="Render DPI (default: 80)")
    p.add_argument("--net-alpha", type=float, default=0.2,
                   help="Net line opacity (default: 0.2)")
    p.add_argument("--no-nets", dest="draw_nets", action="store_false", default=True,
                   help="Disable net lines")
    p.add_argument("--skip-existing", action="store_true", default=False,
                   help="Skip runs that already have a placement.gif")
    p.add_argument("--output-name", default="placement.gif",
                   help="GIF filename written inside each run_NNN directory "
                        "(default: placement.gif)")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel workers (default: 8)")
    return p.parse_args()


def render_run(
    run_dir: Path,
    bench_name: str,
    benchmark,           # pre-loaded Benchmark object
    args,
) -> bool:
    gif_path = run_dir / args.output_name
    if args.skip_existing and gif_path.exists():
        print(f"  skip  {run_dir.relative_to(REPO_ROOT)}  (already exists)")
        return True

    frames_root = run_dir / "frames" / bench_name

    net_edges = None
    if args.draw_nets:
        net_edges_path = frames_root / "net_edges.pt"
        if net_edges_path.exists():
            net_edges = torch.load(net_edges_path, weights_only=False)

    ordered = ftg.build_ordered_frames(frames_root, step=args.step, fps=args.fps)
    if not ordered:
        print(f"  skip  {run_dir.relative_to(REPO_ROOT)}  (no frames after step={args.step})")
        return False

    cfg_names, cfg_n_random, _cfg_n_scatter, _cfg_n_io = ftg._load_visualizer_config()
    highlight_ids = ftg.resolve_highlight_ids(benchmark, cfg_names, cfg_n_random) or None

    pil_frames: list[Image.Image] = []
    frame_durations: list[int]    = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, (frame_data, dur) in enumerate(ordered):
            png_path = tmpdir / f"frame_{i:05d}.png"
            ftg.render_frame(frame_data, benchmark, net_edges,
                             args.net_alpha, str(png_path), args.dpi,
                             highlight_ids=highlight_ids)
            pil_frames.append(Image.open(png_path).copy())
            frame_durations.append(dur)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_durations,
        loop=0,
        optimize=False,
    )
    size_kb = gif_path.stat().st_size / 1024
    print(
        f"  done  {run_dir.parent.name}/{run_dir.name}"
        f"  {len(pil_frames)}f  → {gif_path.relative_to(REPO_ROOT)}"
        f"  ({size_kb:.0f} KB)"
    )
    return True


def main():
    args = parse_args()

    if args.sweep_dir:
        sweep_path = Path(args.sweep_dir)
        if not sweep_path.is_absolute():
            sweep_path = REPO_ROOT / sweep_path
    else:
        sweep_path = find_latest_sweep()
        if sweep_path is None:
            print("[error] No sweep_* directories found in sweep/", file=sys.stderr)
            sys.exit(1)

    if not sweep_path.exists():
        print(f"[error] Sweep directory not found: {sweep_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Sweep : {sweep_path.relative_to(REPO_ROOT)}")

    benchmark_filter = set(args.benchmark) if args.benchmark else None
    runs = find_runs(sweep_path, benchmark_filter)
    if not runs:
        print("[error] No runs with recorded frames found.", file=sys.stderr)
        sys.exit(1)

    print(f"Runs  : {len(runs)}  (step={args.step}, fps={args.fps}, dpi={args.dpi})")
    print()

    # Pre-load each unique benchmark once (serial, cheap relative to rendering).
    bench_names = sorted({bench_name for _, bench_name in runs})
    benchmark_cache: dict[str, object] = {}
    for bench_name in bench_names:
        try:
            benchmark_cache[bench_name] = ftg.load_benchmark_for_frames(bench_name)
        except Exception as exc:
            print(f"  ERROR loading benchmark '{bench_name}': {exc}", file=sys.stderr)

    runs = [(rd, bn) for rd, bn in runs if bn in benchmark_cache]

    n_workers = min(args.workers, len(runs)) if runs else 1
    print(f"Workers: {n_workers}\n")
    ok = err = 0

    futures = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for run_dir, bench_name in runs:
            fut = executor.submit(
                render_run, run_dir, bench_name,
                benchmark_cache[bench_name], args,
            )
            futures[fut] = (run_dir, bench_name)

        for fut in as_completed(futures):
            try:
                success = fut.result()
            except Exception as exc:
                run_dir, bench_name = futures[fut]
                print(f"  ERROR {run_dir.name} {bench_name}: {exc}")
                success = False
            if success:
                ok += 1
            else:
                err += 1

    print(f"\nDone: {ok} GIF(s) written, {err} skipped/failed.")


if __name__ == "__main__":
    main()
