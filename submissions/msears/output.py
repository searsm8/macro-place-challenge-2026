"""
output.py — Output utilities for CometPlacer.

Console logging, frame export, iteration CSV log, and per-macro HPWL table.
"""

import os
import torch
from pathlib import Path


def printBenchmarkInfo(benchmark, density_method):
    """Print a one-time summary of the benchmark and placer settings."""
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    num_macros = benchmark.num_macros
    sizes = benchmark.macro_sizes[:num_macros]
    utilisation = (sizes[:, 0] * sizes[:, 1]).sum() / (canvas_w * canvas_h) * 100

    print(f"\n  Benchmark : {benchmark.name}")
    print(f"  Macros    : {benchmark.num_hard_macros} hard + "
          f"{benchmark.num_soft_macros} soft = {num_macros} total  "
          f"({int(benchmark.macro_fixed[:num_macros].sum())} fixed)")
    print(f"  Nets      : {benchmark.num_nets}   "
          f"I/O ports: {benchmark.port_positions.shape[0]}")
    print(f"  Canvas    : {canvas_w:.2f} x {canvas_h:.2f} um   "
          f"Grid: {benchmark.grid_rows}r x {benchmark.grid_cols}c")
    print(f"  Area util : {utilisation:.1f}%   "
          f"W:[{sizes[:,0].min():.3f}, {sizes[:,0].max():.3f}]  "
          f"H:[{sizes[:,1].min():.3f}, {sizes[:,1].max():.3f}]")
    print(f"  Density   : {density_method}")
    print()


def writeMacroDat(pos, benchmark, net_data, plc, out_path):
    """
    Compute exact HPWL per net at final positions, attribute each net's
    HPWL to every connected hard macro, and write a sorted table.

    Columns: macro_name  width  height  connected_nets  total_hpwl
    Rows sorted descending by total_hpwl.
    """
    num_hard = benchmark.num_hard_macros
    macro_ids = net_data["macro_ids"]
    offsets = net_data["offsets"]
    net_ids = net_data["net_ids"]
    is_macro = net_data["is_macro"]
    num_nets = net_data["num_nets"]

    with torch.no_grad():
        safe_ids = macro_ids.clamp(min=0)
        macro_flag = is_macro.float().unsqueeze(1)
        pin_pos = pos[safe_ids] * macro_flag + offsets

        # Exact HPWL per net
        max_xy = pin_pos.new_full((num_nets, 2), float("-inf"))
        min_xy = pin_pos.new_full((num_nets, 2), float("inf"))
        idx2 = net_ids.unsqueeze(1).expand(-1, 2)
        max_xy.scatter_reduce_(0, idx2, pin_pos, reduce="amax", include_self=True)
        min_xy.scatter_reduce_(0, idx2, pin_pos, reduce="amin", include_self=True)
        net_hpwl = (max_xy - min_xy).sum(dim=1)

        # Attribute nets to hard macros (deduplicated)
        hard_mask = is_macro & (macro_ids < num_hard)
        m_ids = macro_ids[hard_mask]
        n_ids = net_ids[hard_mask]

        pairs = torch.stack([m_ids, n_ids], dim=1)
        unique_pairs = torch.unique(pairs, dim=0)

        macro_hpwl = torch.zeros(num_hard)
        macro_net_cnt = torch.zeros(num_hard, dtype=torch.long)
        macro_hpwl.scatter_add_(0, unique_pairs[:, 0], net_hpwl[unique_pairs[:, 1]])
        macro_net_cnt.scatter_add_(0, unique_pairs[:, 0],
                                   torch.ones(len(unique_pairs), dtype=torch.long))

    names = [plc.modules_w_pins[plc_idx].get_name()
             for plc_idx in plc.hard_macro_indices]
    sizes = benchmark.macro_sizes[:num_hard]
    widths = sizes[:, 0].tolist()
    heights = sizes[:, 1].tolist()
    hpwl_vals = macro_hpwl.tolist()
    net_cnts = macro_net_cnt.tolist()

    order = sorted(range(num_hard), key=lambda i: hpwl_vals[i], reverse=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f"{'macro_name':<40}  {'width':>10}  {'height':>10}  "
                 f"{'nets':>6}  {'total_hpwl':>14}\n")
        fh.write("-" * 86 + "\n")
        for i in order:
            fh.write(f"{names[i]:<40}  {widths[i]:>10.4f}  {heights[i]:>10.4f}  "
                     f"{int(net_cnts[i]):>6}  {hpwl_vals[i]:>14.6f}\n")


class OutputManager:
    """
    Manages all placer output: console logging, per-iteration CSV,
    frame snapshots for animation.
    """

    def __init__(self, output_cfg):
        self.log_every = output_cfg.get("log_every", 50)
        self.record_frames = output_cfg.get("record_frames", False)
        self.record_iters = output_cfg.get("record_iterations", True)
        self.frames_dir = output_cfg.get("frames_dir", "vis/frames")
        self.quiet = output_cfg.get("quiet", False)

        if os.environ.get("MSPLACER_NO_FRAMES") == "1":
            self.record_frames = False
        if frames_env := os.environ.get("MSPLACER_FRAMES_DIR"):
            self.frames_dir = frames_env
            self.record_frames = True

        self._iter_fh = None
        self._bench_frames_dir = None

    def log(self, msg, **kwargs):
        """Print unless quiet mode is active."""
        if not self.quiet:
            print(msg, **kwargs)

    def setupFrames(self, benchmark, net_data):
        """Create frame directory and save net topology snapshot."""
        if not self.record_frames:
            return
        frames_dir = Path(self.frames_dir) / benchmark.name
        if frames_dir.exists():
            for f in frames_dir.glob("frame_*.pt"):
                f.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"  Recording frames -> {frames_dir}/")

        num_nets = net_data["num_nets"]
        net_ids_t = net_data["net_ids"]
        _, counts = torch.unique_consecutive(net_ids_t, return_counts=True)
        first_pin = torch.zeros(num_nets, dtype=torch.long)
        if num_nets > 1:
            first_pin[1:] = counts[:-1].cumsum(0)
        torch.save({
            "macro_ids": net_data["macro_ids"],
            "offsets": net_data["offsets"],
            "net_ids": net_ids_t,
            "is_macro": net_data["is_macro"],
            "num_nets": num_nets,
            "first_pin_idx": first_pin,
        }, frames_dir / "net_edges.pt")
        self.log(f"  Net topology    -> {frames_dir}/net_edges.pt")
        self._bench_frames_dir = frames_dir

    def openIterLog(self, benchmark):
        """Open iterations.dat for writing per-iteration metrics."""
        if not self.record_iters:
            return
        iter_path = Path(self.frames_dir).parent / "data" / benchmark.name / "iterations.dat"
        iter_path.parent.mkdir(parents=True, exist_ok=True)
        self._iter_fh = open(iter_path, "w")
        self._iter_fh.write("Iter, HPWL, OVFW, alpha, lambda, gamma\n")
        self.log(f"  Iteration log   -> {iter_path}")

    def writeIter(self, iter_num, wl_loss, overflow, alpha, lambda_d, gamma):
        """Write one row to iterations.dat."""
        if self._iter_fh is not None:
            self._iter_fh.write(
                f"{iter_num+1:04d}, {wl_loss:.4e}, {overflow:.4e}, "
                f"{alpha:.4e}, {lambda_d:.4e}, {gamma:.4e}\n")

    def shouldLog(self, iter_num, max_iters):
        """Return True if this iteration should print a progress line."""
        return (self.log_every > 0
                and (iter_num % self.log_every == 0 or iter_num == max_iters - 1))

    def saveFrame(self, iter_num, pos, wl_val, den_energy, overflow,
                  lambda_d, alpha, gamma, benchmark, num_macros):
        """Save a frame snapshot to disk."""
        if not self.record_frames or self._bench_frames_dir is None:
            return
        torch.save({
            "iter": iter_num,
            "positions": pos.detach().clone(),
            "wl_loss": wl_val,
            "den_loss": den_energy,
            "overflow": overflow,
            "lambda_d": lambda_d,
            "alpha": alpha,
            "gamma": gamma,
            "benchmark_name": benchmark.name,
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
            "macro_sizes": benchmark.macro_sizes[:num_macros].clone(),
            "num_hard": benchmark.num_hard_macros,
        }, self._bench_frames_dir / f"frame_{iter_num:05d}.pt")

    def saveMipFrame(self, pos, benchmark, num_macros):
        """Save the quadratic-init (mIP) placement as frame_mip.pt."""
        if not self.record_frames or self._bench_frames_dir is None:
            return
        torch.save({
            "iter": -1,
            "positions": pos.detach().clone(),
            "wl_loss": 0.0,
            "den_loss": 0.0,
            "overflow": float("inf"),
            "lambda_d": 0.0,
            "alpha": 0.0,
            "gamma": 0.0,
            "benchmark_name": benchmark.name,
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
            "macro_sizes": benchmark.macro_sizes[:num_macros].clone(),
            "num_hard": benchmark.num_hard_macros,
        }, self._bench_frames_dir / "frame_mip.pt")

    def saveLegalFrame(self, iter_num, pos, wl_val, gamma, benchmark, num_macros):
        """Save the legalized placement as frame_legal.pt."""
        if not self.record_frames or self._bench_frames_dir is None:
            return
        torch.save({
            "iter": iter_num,
            "positions": pos.detach().clone(),
            "wl_loss": wl_val,
            "den_loss": 0.0,
            "overflow": 0.0,
            "lambda_d": 0.0,
            "alpha": 0.0,
            "gamma": gamma,
            "benchmark_name": benchmark.name,
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
            "macro_sizes": benchmark.macro_sizes[:num_macros].clone(),
            "num_hard": benchmark.num_hard_macros,
        }, self._bench_frames_dir / "frame_legal.pt")

    def close(self):
        """Close the iteration log file."""
        if self._iter_fh is not None:
            self._iter_fh.close()
            self._iter_fh = None
