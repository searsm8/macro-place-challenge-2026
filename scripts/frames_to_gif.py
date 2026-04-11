"""
frames_to_gif.py — Convert MSPlacer frame snapshots into an animated GIF.

Loads the per-iteration .pt files written by MSPlacer when record_frames=true
and stitches them into a GIF.  If net_edges.pt is present in the same directory
(also written by MSPlacer), net connections are drawn as a gray LineCollection
on each frame — no PlacementCost (plc) object is needed, so rendering stays fast.

Each frame is rendered as a single-panel matplotlib figure showing:
  • Hard macros (steelblue), soft macros (lightsteelblue), fixed macros (red)
  • I/O port pins (green)
  • Net connections (gray, alpha=0.05) — star topology: driver → each sink

Usage:
    # Default: reads vis/frames/ibm01/, writes vis/ibm01_opt.gif
    uv run python scripts/frames_to_gif.py

    # Specify benchmark and output
    uv run python scripts/frames_to_gif.py --benchmark ibm03 --output vis/ibm03.gif

    # Slower GIF (more time per frame to read)
    uv run python scripts/frames_to_gif.py --fps 10

    # Only render every Nth frame (useful for long runs)
    uv run python scripts/frames_to_gif.py --step 5

    # Disable net lines even if net_edges.pt is present
    uv run python scripts/frames_to_gif.py --no-nets
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Convert MSPlacer frame snapshots to GIF.")
    p.add_argument("--benchmark", "-b", default="ibm01",
                   help="Benchmark name (default: ibm01)")
    p.add_argument("--frames-dir", default="vis/frames",
                   help="Root directory containing per-benchmark frame folders "
                        "(default: vis/frames)")
    p.add_argument("--output", "-o", default=None,
                   help="Output GIF path (default: vis/<benchmark>_opt.gif)")
    p.add_argument("--fps", type=float, default=20,
                   help="Frames per second in the GIF (default: 20)")
    p.add_argument("--step", type=int, default=1,
                   help="Use every Nth frame, e.g. --step 5 (default: 1 = all frames)")
    p.add_argument("--dpi", type=int, default=80,
                   help="Render DPI — lower = smaller/faster (default: 80)")
    p.add_argument("--no-nets", dest="draw_nets", action="store_false", default=True,
                   help="Disable net connection lines even if net_edges.pt is present")
    p.add_argument("--net-alpha", type=float, default=0.05,
                   help="Opacity of net lines (default: 0.05). "
                        "Raise to 0.1–0.2 for sparser benchmarks.")
    return p.parse_args()


def load_benchmark_for_frames(benchmark_name):
    """
    Load the benchmark object so we can access macro_fixed, port_positions, etc.
    Mirrors _load_plc() in placer.py but returns the benchmark, not the plc.
    """
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    from macro_place.benchmark import Benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    if root.exists():
        benchmark, _ = load_benchmark_from_dir(str(root))
        return benchmark

    ng45_map = {
        "ariane133_ng45":    "ariane133",
        "ariane136_ng45":    "ariane136",
        "nvdla_ng45":        "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    design = ng45_map.get(benchmark_name)
    if design:
        base = (Path("external/MacroPlacement/Flows/NanGate45")
                / design / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            benchmark, _ = load_benchmark(
                str(base / "netlist.pb.txt"),
                str(base / "initial.plc"),
            )
            return benchmark

    raise FileNotFoundError(f"Could not find benchmark '{benchmark_name}'")


def render_frame(frame_data, benchmark, net_edges, net_alpha, save_path, dpi):
    """
    Render a single frame to a PNG using a custom single-panel matplotlib figure.

    Draws macros, I/O ports, and (optionally) net connections.  Net connections
    are rendered from the saved net_edges topology using the current frame's macro
    positions — no PlacementCost object is needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection

    positions   = frame_data["positions"]     # [N, 2] float tensor (CPU)
    iteration   = frame_data["iter"]
    wl_loss     = frame_data["wl_loss"]
    gamma       = frame_data["gamma"]
    lambda_d    = frame_data.get("lambda_d", None)
    alpha       = frame_data.get("alpha", None)
    canvas_w    = frame_data["canvas_width"]
    canvas_h    = frame_data["canvas_height"]
    macro_sizes = frame_data["macro_sizes"]   # [N, 2]
    num_hard    = frame_data["num_hard"]
    name        = frame_data["benchmark_name"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), canvas_w, canvas_h,
                            fill=False, edgecolor="black", linewidth=1.5))

    # ── Net connections (drawn first, underneath macros) ──────────────────
    if net_edges is not None:
        macro_ids  = net_edges["macro_ids"]      # [P]
        offsets    = net_edges["offsets"]        # [P, 2]
        net_ids_t  = net_edges["net_ids"]        # [P]
        is_macro   = net_edges["is_macro"]       # [P] bool
        first_pidx = net_edges["first_pin_idx"]  # [K]

        # Absolute pin positions:
        #   macro pin:  macro_center + pin_offset   (differentiable through positions)
        #   port pin:   absolute (x, y) stored in offsets (macro_flag=0)
        safe_ids  = macro_ids.clamp(min=0)
        mflag     = is_macro.float().unsqueeze(1)
        pin_pos   = positions[safe_ids] * mflag + offsets   # [P, 2]

        # Star topology: first pin (driver) connects to every other pin in the net.
        # first_pidx[k] = flat index of the first pin of net k.
        first_pos = pin_pos[first_pidx[net_ids_t]]          # [P, 2]
        is_first  = torch.zeros(len(net_ids_t), dtype=torch.bool)
        is_first[first_pidx] = True
        non_first = ~is_first

        segs = np.stack(
            [first_pos[non_first].numpy(), pin_pos[non_first].numpy()],
            axis=1
        )  # [S, 2, 2]

        ax.add_collection(LineCollection(
            segs, colors="black", alpha=net_alpha, linewidths=0.3, zorder=1
        ))

    # ── Macro rectangles ──────────────────────────────────────────────────
    n = len(positions)
    for i in range(n):
        x, y = positions[i].tolist()
        w, h = macro_sizes[i].tolist()
        is_soft  = i >= num_hard
        is_fixed = benchmark.macro_fixed[i].item()
        color    = "red" if is_fixed else "mediumseagreen" if is_soft else "steelblue"
        alpha    = 0.25 if is_soft else 0.5
        ax.add_patch(Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor=color, edgecolor="black",
            alpha=alpha, linewidth=0.3, zorder=2,
        ))

    # ── I/O ports ─────────────────────────────────────────────────────────
    if benchmark.port_positions.shape[0] > 0:
        ax.scatter(
            benchmark.port_positions[:, 0].numpy(),
            benchmark.port_positions[:, 1].numpy(),
            s=6, c="green", zorder=5,
        )

    # ── Title with per-frame metadata ─────────────────────────────────────
    nets_label   = f"  {net_edges['num_nets']}nets" if net_edges is not None else ""
    alpha_label  = f"   α={alpha:.4f}" if alpha is not None else ""
    lambda_label = f"   λ={lambda_d:.2e}" if lambda_d is not None else ""
    ax.set_title(
        f"{name}{nets_label}  |  iter {iteration:4d}   "
        f"WL={wl_loss:.1f}   γ={gamma:.4f}{alpha_label}{lambda_label}",
        fontsize=9,
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    frames_root = Path(args.frames_dir) / args.benchmark
    if not frames_root.exists():
        print(f"[error] Frame directory not found: {frames_root}", file=sys.stderr)
        print("  Run the placer with record_frames = true in config.toml first.",
              file=sys.stderr)
        sys.exit(1)

    frame_files = sorted(frames_root.glob("frame_*.pt"))
    if not frame_files:
        print(f"[error] No frame_*.pt files found in {frames_root}", file=sys.stderr)
        sys.exit(1)

    # Apply step subsampling
    frame_files = frame_files[::args.step]
    n_frames = len(frame_files)

    output_path = Path(args.output) if args.output else Path(f"vis/{args.benchmark}_opt.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load net topology (written once by the placer alongside the frames)
    net_edges = None
    if args.draw_nets:
        net_edges_path = frames_root / "net_edges.pt"
        if net_edges_path.exists():
            print(f"  Net edges : {net_edges_path}")
            net_edges = torch.load(net_edges_path, weights_only=False)
        else:
            print("  Net edges : not found — nets will not be drawn")
            print("    (run placer with record_frames = true to generate net_edges.pt)")

    print(f"  Benchmark : {args.benchmark}")
    print(f"  Frames    : {n_frames}  (step={args.step})")
    print(f"  FPS       : {args.fps}")
    print(f"  Output    : {output_path}")
    if net_edges is not None:
        print(f"  Nets      : {net_edges['num_nets']}  alpha={args.net_alpha}")
    print()

    # Load the benchmark once (for macro_fixed, port_positions, etc.)
    print("  Loading benchmark...", end=" ", flush=True)
    benchmark = load_benchmark_for_frames(args.benchmark)
    print("done")

    # Render each frame to a temporary PNG, collect PIL Images
    pil_frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fpath in enumerate(frame_files):
            frame_data = torch.load(fpath, weights_only=False)
            png_path = Path(tmpdir) / f"frame_{i:05d}.png"

            render_frame(frame_data, benchmark, net_edges, args.net_alpha,
                         str(png_path), args.dpi)

            pil_frames.append(Image.open(png_path).copy())  # copy before tmpdir cleanup

            # Progress bar
            pct = (i + 1) / n_frames * 100
            bar = "#" * int(pct / 2)
            print(f"\r  Rendering [{bar:<50}] {pct:5.1f}%  ({i+1}/{n_frames})",
                  end="", flush=True)

    print()  # newline after progress bar

    # Stitch into GIF with PIL
    duration_ms = int(1000 / args.fps)
    print(f"  Writing GIF ({duration_ms}ms/frame)...", end=" ", flush=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,           # 0 = loop forever
        optimize=False,   # skip optimisation pass — much faster for large GIFs
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"done  ({size_kb:.0f} KB)")
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
