"""
frames_to_gif.py — Convert CometPlacer frame snapshots into an animated GIF.

Loads the per-iteration .pt files written by CometPlacer when record_frames=true
and stitches them into a GIF.  If net_edges.pt is present in the same directory
(also written by CometPlacer), net connections are drawn as a gray LineCollection
on each frame — no PlacementCost (plc) object is needed, so rendering stays fast.

Each frame is rendered as a single-panel matplotlib figure showing:
  • Hard macros (steelblue), soft macros (lightsteelblue), fixed macros (red)
  • I/O port pins (green)
  • Net connections (gray, alpha=0.1) — star topology: driver → each sink

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


def _load_visualizer_config():
    """
    Read highlight settings from submissions/msears/config.toml.
    Returns (highlight_macros: list[str], highlight_random_macros: int, highlight_scatter_macros: int).
    Falls back to ([], 0, 0) if the key / file is absent.
    """
    import tomllib
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "submissions" / "msears" / "config.toml"
    if not cfg_path.exists():
        return [], 0, 0
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    out = cfg.get("output", {})
    names     = list(out.get("highlight_macros", []))
    n_rand    = int(out.get("highlight_random_macros", 0))
    n_scatter = int(out.get("highlight_scatter_macros", 0))
    return names, n_rand, n_scatter


def _sample_highlight_ids(benchmark, n):
    """
    Return a set of n macro indices sampled without replacement, weighted by
    macro area (w × h).  Fixed macros are excluded from sampling.
    """
    import random
    sizes = benchmark.macro_sizes          # [N, 2] tensor
    fixed = benchmark.macro_fixed          # [N] bool tensor
    areas = (sizes[:, 0] * sizes[:, 1]).tolist()
    candidates = [i for i, f in enumerate(fixed.tolist()) if not f]
    weights    = [areas[i] for i in candidates]
    k = min(n, len(candidates))
    chosen = random.choices(candidates, weights=weights, k=k * 10)  # oversample, dedup
    seen, result = set(), []
    for idx in chosen:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
        if len(result) == k:
            break
    return set(result)


def resolve_highlight_ids(benchmark, named_macros=(), n_random=0):
    """
    Resolve macro names / random count to a set of benchmark macro indices.
    Named macros take priority; falls back to random area-weighted sampling.
    """
    name_to_idx = {name: i for i, name in enumerate(benchmark.macro_names)}
    highlight_ids = set()
    for macro_name in named_macros:
        if macro_name in name_to_idx:
            highlight_ids.add(name_to_idx[macro_name])
    if not highlight_ids and n_random > 0:
        highlight_ids = _sample_highlight_ids(benchmark, n_random)
    return highlight_ids


def _annotate_macros_dat(dat_path, highlight_names):
    """
    Rewrite macros.dat in-place, appending '*' to the name of each highlighted
    macro.  Idempotent: existing '*' markers are stripped first so re-running
    doesn't keep adding them.
    """
    lines = dat_path.read_text().splitlines()
    out = []
    for line in lines:
        # Header / separator lines pass through unchanged
        if not line or line.startswith("-") or line.startswith("macro_name"):
            out.append(line)
            continue
        # Name occupies the first 40 characters (left-aligned, space-padded)
        raw_name = line[:40].rstrip().rstrip("*")  # strip any prior '*'
        rest     = line[40:]
        marker   = "*" if raw_name in highlight_names else " "
        out.append(f"{raw_name + marker:<40}{rest}")
    dat_path.write_text("\n".join(out) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Convert CometPlacer frame snapshots to GIF.")
    p.add_argument("--benchmark", "-b", default="ibm01",
                   help="Benchmark name (default: ibm01)")
    p.add_argument("--frames-dir", default="vis/frames",
                   help="Root directory containing per-benchmark frame folders "
                        "(default: vis/frames)")
    p.add_argument("--output", "-o", default=None,
                   help="Output GIF path (default: vis/<benchmark>_opt.gif)")
    p.add_argument("--fps", type=float, default=10,
                   help="Frames per second in the GIF (default: 20)")
    p.add_argument("--step", type=int, default=10,
                   help="Use every Nth frame, e.g. --step 5 (default: 10)")
    p.add_argument("--dpi", type=int, default=80,
                   help="Render DPI — lower = smaller/faster (default: 80)")
    p.add_argument("--no-nets", dest="draw_nets", action="store_false", default=True,
                   help="Disable net connection lines even if net_edges.pt is present")
    p.add_argument("--net-alpha", type=float, default=0.1,
                   help="Opacity of net lines (default: 0.1). "
                        "Raise to 0.1–0.2 for sparser benchmarks.")
    p.add_argument("--highlight", nargs="+", default=[], metavar="MACRO_NAME",
                   help="Macro names whose nets are drawn at alpha=0.95 "
                        "(e.g. --highlight macro_0 macro_3)")
    p.add_argument("--highlight-random", type=int, default=None, metavar="N",
                   help="Randomly highlight N macros, weighted by area (overrides "
                        "highlight_random_macros from config.toml). 0 = disabled.")
    p.add_argument("--legal-only", action="store_true", default=False,
                   help="Only render the legalized frame as a PNG (skip GIF)")
    p.add_argument("--mip-only", action="store_true", default=False,
                   help="Only render the quadratic-init (mIP) frame as a PNG (skip GIF)")
    return p.parse_args()


def build_ordered_frames(frames_root, step=1, fps=10, pause_ms=5000):
    """
    Build the ordered (frame_data, duration_ms) sequence for a single run.

    Ordering:
      mIP frame (5 s)  →  phases 2/2.5 (pause at each phase boundary)
      →  legal frame (5 s)  →  phase 4 (pause at boundary)

    Returns an empty list if no numbered frames are found.
    """
    from pathlib import Path as _Path
    frames_root = _Path(frames_root)
    duration_ms = int(1000 / fps)

    mip_path   = frames_root / "frame_mip.pt"
    legal_path = frames_root / "frame_legal.pt"

    num_files = sorted(frames_root.glob("frame_[0-9]*.pt"))[::step]
    if not num_files:
        return []

    num_data     = [torch.load(f, weights_only=False) for f in num_files]
    frame_phases = [d.get("phase", "") for d in num_data]

    split_idx   = next((i for i, p in enumerate(frame_phases) if "4:" in p), len(frame_phases))
    pre_list    = list(zip(num_data[:split_idx],  frame_phases[:split_idx]))
    phase4_list = list(zip(num_data[split_idx:],  frame_phases[split_idx:]))

    ordered = []

    if mip_path.exists():
        mip_data = torch.load(mip_path, weights_only=False)
        mip_data["phase"] = "1: mIP"  # override in case old runs saved without phase field
        ordered.append((mip_data, pause_ms))

    for i, (data, phase) in enumerate(pre_list):
        next_phase  = pre_list[i + 1][1] if i + 1 < len(pre_list) else None
        at_boundary = next_phase is None or next_phase != phase
        ordered.append((data, pause_ms if at_boundary else duration_ms))

    if legal_path.exists():
        legal_data = torch.load(legal_path, weights_only=False)
        legal_data["phase"] = "3: mLG"  # override in case old runs saved wrong label
        ordered.append((legal_data, pause_ms))

    for i, (data, phase) in enumerate(phase4_list):
        next_phase  = phase4_list[i + 1][1] if i + 1 < len(phase4_list) else None
        at_boundary = next_phase is None or next_phase != phase
        ordered.append((data, pause_ms if at_boundary else duration_ms))

    return ordered


def load_benchmark_for_frames(benchmark_name):
    """
    Load the benchmark object so we can access macro_fixed, port_positions, etc.
    Mirrors _load_plc() in placer.py but returns the benchmark, not the plc.
    """
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    from macro_place.benchmark import Benchmark

    # Anchor to the repo root (two levels up from scripts/) so this works
    # regardless of the CWD the script is invoked from.
    repo_root = Path(__file__).resolve().parent.parent

    root = repo_root / "external/MacroPlacement/Testcases/ICCAD04" / benchmark_name
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
        base = (repo_root / "external/MacroPlacement/Flows/NanGate45"
                / design / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            benchmark, _ = load_benchmark(
                str(base / "netlist.pb.txt"),
                str(base / "initial.plc"),
            )
            return benchmark

    raise FileNotFoundError(f"Could not find benchmark '{benchmark_name}'")


def render_frame(frame_data, benchmark, net_edges, net_alpha, save_path, dpi,
                 highlight_ids=None, label_override=None):
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
    overflow    = frame_data.get("overflow", None)
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

    # ── Pre-compute pin positions (reused for nets and pin dots) ─────────
    pin_pos = None
    if net_edges is not None:
        macro_ids  = net_edges["macro_ids"]      # [P]
        offsets    = net_edges["offsets"]        # [P, 2]
        net_ids_t  = net_edges["net_ids"]        # [P]
        is_macro   = net_edges["is_macro"]       # [P] bool
        first_pidx = net_edges["first_pin_idx"]  # [K]

        safe_ids = macro_ids.clamp(min=0)
        mflag    = is_macro.float().unsqueeze(1)
        pin_pos  = positions[safe_ids] * mflag + offsets   # [P, 2]

    # ── Net connections (drawn first, underneath macros) ──────────────────
    if pin_pos is not None:
        # Star topology: driver pin → every other pin in the net
        first_pos = pin_pos[first_pidx[net_ids_t]]          # [P, 2]
        is_first  = torch.zeros(len(net_ids_t), dtype=torch.bool)
        is_first[first_pidx] = True
        non_first = ~is_first

        segs = np.stack(
            [first_pos[non_first].numpy(), pin_pos[non_first].numpy()],
            axis=1,
        )  # [S, 2, 2]

        if highlight_ids:
            # A net is "highlighted" if any of its pins belongs to a highlighted macro.
            pin_highlighted = torch.zeros(len(macro_ids), dtype=torch.bool)
            for hid in highlight_ids:
                pin_highlighted |= (macro_ids == hid)
            # Per-net flag: True if any pin in that net is highlighted
            net_highlighted = torch.zeros(net_edges["num_nets"], dtype=torch.float32)
            net_highlighted.scatter_reduce_(
                0, net_ids_t, pin_highlighted.float(),
                reduce="amax", include_self=True,
            )
            net_highlighted = net_highlighted.bool()
            # Map back to the non-first (edge) pins
            nf_net_ids = net_ids_t[non_first]
            edge_hi = net_highlighted[nf_net_ids].numpy()

            ax.add_collection(LineCollection(
                segs[~edge_hi], colors="black", alpha=net_alpha,
                linewidths=0.3, zorder=1,
            ))
            if edge_hi.any():
                ax.add_collection(LineCollection(
                    segs[edge_hi], colors="crimson", alpha=0.95,
                    linewidths=0.8, zorder=2,
                ))
        else:
            ax.add_collection(LineCollection(
                segs, colors="black", alpha=net_alpha, linewidths=0.3, zorder=1,
            ))

    # ── Macro rectangles ──────────────────────────────────────────────────
    n = len(positions)
    for i in range(n):
        x, y = positions[i].tolist()
        w, h = macro_sizes[i].tolist()
        is_soft      = i >= num_hard
        is_fixed     = benchmark.macro_fixed[i].item()
        is_highlight = highlight_ids and (i in highlight_ids)
        if is_highlight:
            color      = "crimson"
            rect_alpha = 0.85
            lw         = 1.2
        elif is_fixed:
            color      = "red"
            rect_alpha = 0.5
            lw         = 0.3
        elif is_soft:
            color      = "mediumseagreen"
            rect_alpha = 0.25
            lw         = 0.3
        else:
            color      = "steelblue"
            rect_alpha = 0.5
            lw         = 0.3
        ax.add_patch(Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor=color, edgecolor="black",
            alpha=rect_alpha, linewidth=lw, zorder=3 if is_highlight else 2,
        ))

    # ── Macro pins (dots on macro faces) — only for highlighted macros ────
    if pin_pos is not None and highlight_ids:
        pin_on_highlight = torch.zeros(len(macro_ids), dtype=torch.bool)
        for hid in highlight_ids:
            pin_on_highlight |= (macro_ids == hid)
        pin_on_highlight &= is_macro
        if pin_on_highlight.any():
            mp = pin_pos[pin_on_highlight].numpy()
            ax.scatter(
                mp[:, 0], mp[:, 1],
                s=4, c="white", linewidths=0.5,
                edgecolors="dimgray", zorder=4, marker="o",
            )

    # ── I/O ports (border pins) ────────────────────────────────────────────
    if benchmark.port_positions.shape[0] > 0:
        ax.scatter(
            benchmark.port_positions[:, 0].numpy(),
            benchmark.port_positions[:, 1].numpy(),
            s=10, c="limegreen", linewidths=0.5,
            edgecolors="darkgreen", zorder=5, marker="D",
        )

    # ── Title with per-frame metadata ─────────────────────────────────────
    nets_label   = f"  {net_edges['num_nets']}nets" if net_edges is not None else ""
    alpha_label  = f"   α={alpha:.4f}" if alpha is not None else ""
    lambda_label = f"   λ={lambda_d:.2e}" if lambda_d is not None else ""
    if overflow is None:
        ovfw_str = ""
    elif overflow == float("inf"):
        ovfw_str = "   ovfw=∞"
    else:
        ovfw_str = f"   ovfw={overflow:.3f}"
    iter_str = f"iter {iteration:4d}"
    ax.set_title(
        f"{name}{nets_label}  |  {iter_str}   "
        f"WL={wl_loss:.1f}{ovfw_str}   γ={gamma:.4f}{alpha_label}{lambda_label}",
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=7, labelbottom=False, labelleft=False)

    # ── Phase label at the bottom ──────────────────────────────────────────
    phase = frame_data.get("phase", "")
    if phase:
        phase_text = f"Phase {phase}"
    elif label_override:
        phase_text = label_override
    else:
        phase_text = ""
    if phase_text:
        ax.text(
            0.5, -0.02, phase_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
        )

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

    # Special frame paths (may or may not exist)
    legal_frame_path = frames_root / "frame_legal.pt"
    mip_frame_path   = frames_root / "frame_mip.pt"
    has_legal = legal_frame_path.exists()
    has_mip   = mip_frame_path.exists()

    output_path = Path(args.output) if args.output else Path(f"vis/{args.benchmark}_opt.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the benchmark once (for macro_fixed, port_positions, etc.)
    print("  Loading benchmark...", end=" ", flush=True)
    benchmark = load_benchmark_for_frames(args.benchmark)
    print("done")

    # Resolve highlight macro names → indices.
    # Priority: --highlight CLI > highlight_macros config > scatter > random sampling.
    cfg_names, cfg_n_random, cfg_n_scatter = _load_visualizer_config()
    name_to_idx = {name: i for i, name in enumerate(benchmark.macro_names)}

    highlight_ids = set()
    named_sources = args.highlight or cfg_names   # CLI takes priority over config
    if named_sources:
        for macro_name in named_sources:
            if macro_name in name_to_idx:
                highlight_ids.add(name_to_idx[macro_name])
            else:
                print(f"  [warn] highlight: macro '{macro_name}' not found, skipping")
        if highlight_ids:
            found = [benchmark.macro_names[i] for i in sorted(highlight_ids)]
            src = "CLI" if args.highlight else "config"
            print(f"  Highlight : {found}  (ids: {sorted(highlight_ids)})  [{src}]")

    if not highlight_ids and cfg_n_scatter > 0:
        scatter_pt = frames_root / "scatter_ids.pt"
        if scatter_pt.exists():
            import torch as _torch
            data = _torch.load(scatter_pt, weights_only=True)
            ids = data["scatter_ids"].tolist()[:cfg_n_scatter]
            highlight_ids = set(ids)
            found = [benchmark.macro_names[i] for i in ids]
            print(f"  Highlight : {found}  (ids: {ids})  [scatter, n={cfg_n_scatter}]")

    if not highlight_ids:
        n_random = args.highlight_random if args.highlight_random is not None \
                   else cfg_n_random
        if n_random > 0:
            highlight_ids = _sample_highlight_ids(benchmark, n_random)
            found = [benchmark.macro_names[i] for i in sorted(highlight_ids)]
            print(f"  Highlight : {found}  (ids: {sorted(highlight_ids)})  [random, n={n_random}]")

    # Annotate macros.dat with '*' next to highlighted macros (if it exists).
    if highlight_ids:
        highlight_names = {benchmark.macro_names[i] for i in highlight_ids}
        macros_dat = frames_root / "macros.dat"
        if macros_dat.exists():
            _annotate_macros_dat(macros_dat, highlight_names)
            print(f"  macros.dat: annotated with '*' for highlighted macros")

    # --mip-only: render just the mIP (quadratic-init) PNG and exit
    if args.mip_only:
        if not mip_frame_path.exists():
            print(f"[error] No frame_mip.pt found in {frames_root}", file=sys.stderr)
            print("  Run the placer with initial_placement = \"quadratic\" and "
                  "record_frames = true.", file=sys.stderr)
            sys.exit(1)
        mip_png_path = (Path(args.output) if args.output
                        else Path(f"vis/{args.benchmark}_mip.png"))
        mip_png_path.parent.mkdir(parents=True, exist_ok=True)
        frame_data = torch.load(mip_frame_path, weights_only=False)
        render_frame(frame_data, benchmark, net_edges, args.net_alpha,
                     str(mip_png_path), dpi=150,
                     highlight_ids=highlight_ids or None,
                     label_override="mIP (quadratic init)")
        print(f"  Saved mIP image: {mip_png_path}")
        return

    # --legal-only: render just the legalized PNG and exit
    if args.legal_only:
        if not has_legal:
            print(f"[error] No frame_legal.pt found in {frames_root}", file=sys.stderr)
            sys.exit(1)
        legal_png_path = (Path(args.output) if args.output
                          else Path(f"vis/{args.benchmark}_legal.png"))
        legal_png_path.parent.mkdir(parents=True, exist_ok=True)
        frame_data = torch.load(legal_frame_path, weights_only=False)
        render_frame(frame_data, benchmark, net_edges, args.net_alpha,
                     str(legal_png_path), dpi=150,
                     highlight_ids=highlight_ids or None)
        print(f"  Saved legalized image: {legal_png_path}")
        return

    # -- Build ordered sequence -----------------------------------------------
    duration_ms = int(1000 / args.fps)
    print("  Scanning phases...", end=" ", flush=True)
    ordered = build_ordered_frames(frames_root, step=args.step, fps=args.fps)
    if not ordered:
        print(f"\n[error] No numbered frame_*.pt files found in {frames_root}", file=sys.stderr)
        sys.exit(1)
    print(f"done  ({len(ordered)} entries)")

    n_frames = len(ordered)
    print(f"  Benchmark : {args.benchmark}")
    print(f"  Frames    : {n_frames}  (step={args.step})")
    print(f"  FPS       : {args.fps}")
    print(f"  Output    : {output_path}")
    if net_edges is not None:
        print(f"  Nets      : {net_edges['num_nets']}  alpha={args.net_alpha}")
    print(f"  mIP frame : {'yes (phase 1, 5 s pause)' if has_mip else 'not found — skipped'}")
    print(f"  Legal frame: {'yes (phase 3, between ph2 and ph4, 5 s pause)' if has_legal else 'not found — skipped'}")
    print(f"  Total      : {n_frames} entries (incl. phase pauses)")
    print()

    # -- Render each entry to a temporary PNG, collect PIL Images -------------
    pil_frames      = []
    frame_durations = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (frame_data, dur) in enumerate(ordered):
            png_path = Path(tmpdir) / f"frame_{i:05d}.png"
            render_frame(frame_data, benchmark, net_edges, args.net_alpha,
                         str(png_path), args.dpi,
                         highlight_ids=highlight_ids or None)
            pil_frames.append(Image.open(png_path).copy())
            frame_durations.append(dur)

            pct = (i + 1) / n_frames * 100
            bar = "#" * int(pct / 2)
            print(f"\r  Rendering [{bar:<50}] {pct:5.1f}%  ({i+1}/{n_frames})",
                  end="", flush=True)

        # Standalone high-res PNG of the legalized design
        if has_legal:
            legal_png_path = output_path.with_name(
                output_path.stem.replace("_opt", "") + "_legal.png"
            )
            render_frame(torch.load(legal_frame_path, weights_only=False),
                         benchmark, net_edges, args.net_alpha,
                         str(legal_png_path), dpi=150,
                         highlight_ids=highlight_ids or None)
            print(f"\n  Saved legalized image: {legal_png_path}")

    print()  # newline after progress bar

    # -- Stitch into GIF with PIL ---------------------------------------------
    print(f"  Writing GIF ({duration_ms}ms/frame)...", end=" ", flush=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_durations,
        loop=0,           # 0 = loop forever
        optimize=False,   # skip optimisation pass — much faster for large GIFs
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"done  ({size_kb:.0f} KB)")
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
