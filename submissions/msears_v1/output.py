"""
Output utilities for MSPlacer: console logging, frame export, iteration log,
and per-macro HPWL table.
"""

import os
import torch
from pathlib import Path


def print_benchmark_info(benchmark, density_method, hpwl_gradient_method):
    """Print a one-time summary of the benchmark and placer settings."""
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    n = benchmark.num_macros
    sizes = benchmark.macro_sizes[:n]
    util = (sizes[:, 0] * sizes[:, 1]).sum() / (cw * ch) * 100

    print(f"\n  Benchmark : {benchmark.name}")
    print(f"  Macros    : {benchmark.num_hard_macros} hard + "
          f"{benchmark.num_soft_macros} soft = {n} total  "
          f"({int(benchmark.macro_fixed[:n].sum())} fixed)")
    print(f"  Nets      : {benchmark.num_nets}   "
          f"I/O ports: {benchmark.port_positions.shape[0]}")
    print(f"  Canvas    : {cw:.2f} x {ch:.2f} um   "
          f"Grid: {benchmark.grid_rows}r x {benchmark.grid_cols}c")
    print(f"  Area util : {util:.1f}%   "
          f"W:[{sizes[:,0].min():.3f}, {sizes[:,0].max():.3f}]  "
          f"H:[{sizes[:,1].min():.3f}, {sizes[:,1].max():.3f}]")
    print(f"  Density   : {density_method}  "
          f"HPWL: {hpwl_gradient_method}")
    print()


def write_macro_dat(pos, benchmark, net_data, plc, out_path):
    """
    Compute exact HPWL per net at final positions, attribute each net's HPWL
    to every hard macro that has at least one pin on it, and write a sorted
    table to out_path.

    Columns: macro_name  width  height  connected_nets  total_hpwl
    Rows sorted descending by total_hpwl.
    """
    n_hard    = benchmark.num_hard_macros
    macro_ids = net_data["macro_ids"]   # [P]
    offsets   = net_data["offsets"]     # [P, 2]
    net_ids   = net_data["net_ids"]     # [P]
    is_macro  = net_data["is_macro"]    # [P] bool
    K         = net_data["num_nets"]

    with torch.no_grad():
        safe_ids   = macro_ids.clamp(min=0)
        macro_flag = is_macro.float().unsqueeze(1)
        pin_pos    = pos[safe_ids] * macro_flag + offsets   # [P, 2]

        # Exact HPWL per net: (max_x - min_x) + (max_y - min_y)
        max_xy = pin_pos.new_full((K, 2), float("-inf"))
        min_xy = pin_pos.new_full((K, 2), float("inf"))
        idx2   = net_ids.unsqueeze(1).expand(-1, 2)
        max_xy.scatter_reduce_(0, idx2, pin_pos, reduce="amax", include_self=True)
        min_xy.scatter_reduce_(0, idx2, pin_pos, reduce="amin", include_self=True)
        net_hpwl = (max_xy - min_xy).sum(dim=1)   # [K]

        # For each hard macro, sum HPWL of nets it's connected to (deduplicated).
        macro_pin_mask = is_macro & (macro_ids < n_hard)   # hard-macro pins only
        m_ids = macro_ids[macro_pin_mask]                  # [P_hard]
        n_ids = net_ids[macro_pin_mask]                    # [P_hard]

        # Deduplicate (macro_id, net_id) so each net counts once per macro.
        pairs        = torch.stack([m_ids, n_ids], dim=1)  # [P_hard, 2]
        unique_pairs = torch.unique(pairs, dim=0)           # [U, 2]

        macro_hpwl    = torch.zeros(n_hard)
        macro_net_cnt = torch.zeros(n_hard, dtype=torch.long)
        macro_hpwl.scatter_add_(0, unique_pairs[:, 0], net_hpwl[unique_pairs[:, 1]])
        macro_net_cnt.scatter_add_(0, unique_pairs[:, 0],
                                   torch.ones(len(unique_pairs), dtype=torch.long))

    names   = [plc.modules_w_pins[plc_idx].get_name()
               for plc_idx in plc.hard_macro_indices]
    sizes   = benchmark.macro_sizes[:n_hard]
    widths  = sizes[:, 0].tolist()
    heights = sizes[:, 1].tolist()
    hpwl_vals = macro_hpwl.tolist()
    net_cnts  = macro_net_cnt.tolist()

    order = sorted(range(n_hard), key=lambda i: hpwl_vals[i], reverse=True)

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
    Manages all placer output: console logging, per-iteration CSV log,
    frame snapshots for animation, and per-macro HPWL table.

    Constructed from the [output] section of config.toml.
    """

    def __init__(self, o: dict):
        self.log_every         = o.get("log_every", 50)
        self.record_frames     = o.get("record_frames", False)
        self.record_iterations = o.get("record_iterations", True)
        self.frames_dir        = o.get("frames_dir", "vis/frames")
        self.quiet             = o.get("quiet", False)

        # Env var overrides (used by sweep scripts for speed / redirect)
        if os.environ.get("MSPLACER_NO_FRAMES") == "1":
            self.record_frames = False
        if frames_dir_env := os.environ.get("MSPLACER_FRAMES_DIR"):
            self.frames_dir    = frames_dir_env
            self.record_frames = True

        self._iter_fh    = None
        self._frames_dir = None   # set by setup_frames()

    def log(self, msg, **kwargs):
        """Print unless quiet mode is on."""
        if not self.quiet:
            print(msg, **kwargs)

    def setup_frames(self, benchmark, net_data):
        """
        Create (or clear) the frame directory for this benchmark and save the
        net topology snapshot. No-op if record_frames is False.
        """
        if not self.record_frames:
            return
        frames_dir = Path(self.frames_dir) / benchmark.name
        if frames_dir.exists():
            for f in frames_dir.glob("frame_*.pt"):
                f.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"  Recording frames -> {frames_dir}/")

        # Save net topology once — it doesn't change between iterations.
        K = net_data["num_nets"]
        net_ids_t = net_data["net_ids"]
        _, counts = torch.unique_consecutive(net_ids_t, return_counts=True)
        first_pin_idx = torch.zeros(K, dtype=torch.long)
        if K > 1:
            first_pin_idx[1:] = counts[:-1].cumsum(0)
        torch.save({
            "macro_ids":     net_data["macro_ids"],
            "offsets":       net_data["offsets"],
            "net_ids":       net_ids_t,
            "is_macro":      net_data["is_macro"],
            "num_nets":      K,
            "first_pin_idx": first_pin_idx,
        }, frames_dir / "net_edges.pt")
        self.log(f"  Net topology    -> {frames_dir}/net_edges.pt")
        self._frames_dir = frames_dir

    def open_iter_log(self, benchmark):
        """
        Open iterations.dat for writing per-iteration metrics.
        No-op if record_iterations is False.
        """
        if not self.record_iterations:
            return
        iter_path = Path(self.frames_dir) / benchmark.name / "iterations.dat"
        iter_path.parent.mkdir(parents=True, exist_ok=True)
        self._iter_fh = open(iter_path, "w")
        self._iter_fh.write("Iter, HPWL, OVFW, alpha, lambda, gamma\n")
        self.log(f"  Iteration log   -> {iter_path}")

    def write_iter(self, t, wl, overflow, alpha, lambda_d, gamma):
        """Write one row to iterations.dat."""
        if self._iter_fh is not None:
            self._iter_fh.write(f"{t+1:04d}, {wl:.4e}, {overflow:.4e}, "
                                f"{alpha:.4e}, {lambda_d:.4e}, {gamma:.4e}\n")

    def should_log(self, t, max_iters):
        """Return True if this iteration should print a progress line."""
        return self.log_every > 0 and (t % self.log_every == 0 or t == max_iters - 1)

    def save_frame(self, t, pos, wl_val, den_energy, overflow,
                   lambda_d, alpha, gamma, benchmark, n):
        """Save a frame snapshot to disk. No-op if record_frames is False."""
        if not self.record_frames or self._frames_dir is None:
            return
        torch.save({
            "iter":      t,
            "positions": pos.detach().clone(),
            "wl_loss":   wl_val,
            "den_loss":  den_energy,
            "overflow":  overflow,
            "lambda_d":  lambda_d,
            "alpha":     alpha,
            "gamma":     gamma,
            "benchmark_name": benchmark.name,
            "canvas_width":   float(benchmark.canvas_width),
            "canvas_height":  float(benchmark.canvas_height),
            "macro_sizes":    benchmark.macro_sizes[:n].clone(),
            "num_hard":       benchmark.num_hard_macros,
        }, self._frames_dir / f"frame_{t:05d}.pt")

    def close(self):
        """Close the iteration log file if open."""
        if self._iter_fh is not None:
            self._iter_fh.close()
            self._iter_fh = None
