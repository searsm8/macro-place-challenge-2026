"""
MSPlacer - Analytical WA HPWL + Electrostatic Density Gradient Descent

A PyTorch-based analytical placer inspired by DREAMplace (Lin et al. DAC 2019)
and ePlace (Lu et al. DAC 2015).

Core components:
  1. WA (weighted-average) smooth HPWL - a differentiable approximation to
     half-perimeter wirelength, fully vectorized with scatter_add.
  2. Density spreading - repulsive force to prevent macro overlap.
     Two methods selectable via config.toml:
       "bell"          - bell-shape quadratic kernel (local, autograd)
       "electrostatic" - ePlace Poisson FFT solver (global, analytic)
  3. Gradient descent loop combining WL + density forces.

Not yet implemented (to be added incrementally):
  - Legalization / overlap removal
  - Nesterov momentum / BB step size
  - Genetic algorithm over macro orientations

Usage:
    uv run evaluate submissions/msears/placer.py
    uv run evaluate submissions/msears/placer.py --all
    uv run evaluate submissions/msears/placer.py --vis
"""

import os
import sys
import time
import tomllib
import importlib.util
import torch
import numpy as np
from pathlib import Path

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Import sibling module (density.py in same directory)
# ---------------------------------------------------------------------------

def _import_sibling(name):
    """Import a .py file from the same directory as this file by name."""
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_density   = _import_sibling("density")
_legalizer = _import_sibling("legalizer")
_output    = _import_sibling("output")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path=None):
    """
    Load a TOML config file. Returns an empty dict if the file is not found.

    Discovery order:
      1. Explicit path argument (if given)
      2. MSPLACER_CONFIG environment variable (used by sweep for parallelism)
      3. config.toml in the same directory as placer.py  (default)

    Any key absent from the file falls back to the hardcoded default
    inside MSPlacer.__init__.
    """
    if path is None:
        env_path = os.environ.get("MSPLACER_CONFIG")
        path = Path(env_path) if env_path else Path(__file__).parent / "config.toml"
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------

def _load_plc(name):
    """Load the PlacementCost (plc) object for a benchmark by name."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45_map = {
        "ariane133_ng45":    "ariane133",
        "ariane136_ng45":    "ariane136",
        "nvdla_ng45":        "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    design = ng45_map.get(name)
    if design:
        base = (Path("external/MacroPlacement/Flows/NanGate45")
                / design / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"),
                str(base / "initial.plc"),
            )
            return plc

    return None


# ---------------------------------------------------------------------------
# Net data builder
# ---------------------------------------------------------------------------

def _build_net_data(benchmark, plc):
    """
    Precompute vectorized net connectivity tensors for WA HPWL.

    Returns a dict with flat parallel arrays over all pins P across all nets K:

        macro_ids  LongTensor [P]     benchmark macro index per pin (-1 = I/O port)
        offsets    FloatTensor [P, 2] macro pins: (x_off, y_off) relative to center
                                      port pins:  absolute (x, y) position
        net_ids    LongTensor [P]     net index (0..K-1) per pin
        is_macro   BoolTensor [P]     True for macro pins, False for ports
        num_pins   int P
        num_nets   int K

    bidx follows benchmark ordering:
        hard macros: [0, num_hard_macros)
        soft macros: [num_hard_macros, num_macros)
    """
    if plc is None:
        return None

    n_hard = benchmark.num_hard_macros

    # macro name -> benchmark index
    name_to_bidx = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx
    for i, plc_idx in enumerate(plc.soft_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = n_hard + i

    # pin name -> (bidx, ox, oy)
    pin_to_macro = {}
    for pin_plc_idx in plc.hard_macro_pin_indices + plc.soft_macro_pin_indices:
        pin = plc.modules_w_pins[pin_plc_idx]
        pin_name = pin.get_name()
        macro_name = pin_name.rsplit("/", 1)[0]
        if macro_name in name_to_bidx:
            ox, oy = pin.get_offset()
            pin_to_macro[pin_name] = (name_to_bidx[macro_name], float(ox), float(oy))

    # port name -> absolute (x, y)
    port_positions = {}
    for port_plc_idx in plc.port_indices:
        port = plc.modules_w_pins[port_plc_idx]
        px, py = port.get_pos()
        port_positions[port.get_name()] = (float(px), float(py))

    # Flatten nets into parallel arrays
    macro_ids_list, ox_list, oy_list, net_ids_list = [], [], [], []
    net_idx = 0
    num_skipped = 0

    for driver, sinks in plc.nets.items():
        pins_this_net = []
        for pin_name in [driver] + sinks:
            if pin_name in pin_to_macro:
                bidx, ox, oy = pin_to_macro[pin_name]
                pins_this_net.append((bidx, ox, oy))
            elif pin_name in port_positions:
                px, py = port_positions[pin_name]
                pins_this_net.append((-1, px, py))
        if len(pins_this_net) >= 2:
            for (bidx, ox, oy) in pins_this_net:
                macro_ids_list.append(bidx)
                ox_list.append(ox)
                oy_list.append(oy)
                net_ids_list.append(net_idx)
            net_idx += 1
        else:
            num_skipped += 1

    if net_idx == 0:
        return None

    macro_ids = torch.tensor(macro_ids_list, dtype=torch.long)
    offsets   = torch.tensor(list(zip(ox_list, oy_list)), dtype=torch.float32)
    net_ids   = torch.tensor(net_ids_list, dtype=torch.long)
    is_macro  = (macro_ids >= 0)

    return {
        "macro_ids":   macro_ids,
        "offsets":     offsets,
        "net_ids":     net_ids,
        "is_macro":    is_macro,
        "num_pins":    len(macro_ids_list),
        "num_nets":    net_idx,
        "num_skipped": num_skipped,
    }


# ---------------------------------------------------------------------------
# Vectorized WA HPWL
# ---------------------------------------------------------------------------

def _wa_hpwl(pos, net_data, gamma):
    """
    Fully vectorized smooth WA HPWL using scatter_add - no Python loops over nets.

    For each net k with pins i in S_k at positions u_i:

        WA_max_k  = sum_{i in S_k}[ u_i * exp(u_i/g) ] / sum[ exp(u_i/g) ]   ~= max(u_i)
        WA_min_k  = sum_{i in S_k}[ u_i * exp(-u_i/g) ] / sum[ exp(-u_i/g) ] ~= min(u_i)
        loss      = sum_k [ (WA_max_k - WA_min_k)_x + (WA_max_k - WA_min_k)_y ]

    A global shift is subtracted before exp() for numerical stability.
    This shift cancels in the ratio and does not affect the gradient.

    Args:
        pos:      [M, 2] float tensor, macro centers, requires_grad=True
        net_data: dict from _build_net_data()
        gamma:    smoothness parameter (large = smooth; shrinks toward exact HPWL)

    Returns: scalar WL loss
    """
    macro_ids = net_data["macro_ids"]  # [P]
    offsets   = net_data["offsets"]    # [P, 2]
    net_ids   = net_data["net_ids"]    # [P]
    is_macro  = net_data["is_macro"]   # [P] bool
    K = net_data["num_nets"]

    # Absolute pin positions, differentiable through pos for macro pins.
    #
    # We use a masked combination to handle macros and ports in a single op:
    #   pin_pos = pos[safe_ids] * macro_flag + offsets
    #
    # For macro pins (is_macro=True):   1 * pos[bidx] + offset = pos[bidx] + offset  (correct)
    # For port pins  (is_macro=False):  0 * pos[0]   + offset = offset (absolute pos) (correct)
    #
    # safe_ids replaces -1 (port sentinel) with 0 to avoid index error.
    # The gradient of pos[0] from dummy port entries is 0 (multiplied by 0).
    safe_ids   = macro_ids.clamp(min=0)          # [P]
    macro_flag = is_macro.float().unsqueeze(1)   # [P, 1]
    pin_pos    = pos[safe_ids] * macro_flag + offsets   # [P, 2]

    total = pin_pos.new_zeros(())

    for d in range(2):
        u = pin_pos[:, d]  # [P], differentiable through pos

        # Per-net shifts for numerical stability.
        #
        # A global shift (u.max()) breaks when gamma is small: pins far from the
        # global max get exp((u_i - global_max)/gamma) ≈ exp(-100) = 0, so
        # sum_exp underflows to 0 and wa_max = u_sum / 1e-8 blows up negative.
        #
        # Per-net shift = max(u) within each net, broadcast back to [P].
        # Guarantees exp((u_i - shift_k)/gamma) ∈ (0, 1] for every pin,
        # with at least one pin per net hitting exactly 1 (the max pin).
        # The shift cancels in the ratio and does not affect the gradient.
        with torch.no_grad():
            shift_pos = u.new_full((K,), float("-inf"))
            shift_pos.scatter_reduce_(0, net_ids, u, reduce="amax", include_self=True)
            shift_pos = shift_pos[net_ids]   # [P] — each pin gets its net's max

            shift_neg = u.new_full((K,), float("-inf"))
            shift_neg.scatter_reduce_(0, net_ids, -u, reduce="amax", include_self=True)
            shift_neg = shift_neg[net_ids]   # [P] — each pin gets its net's max(-u)

        # WA max: weights = exp(u/gamma), peaks near the maximum u_i
        exp_pos   = torch.exp((u - shift_pos) / gamma)                        # [P]
        sum_exp   = u.new_zeros(K).scatter_add(0, net_ids, exp_pos)           # [K]
        sum_u_exp = u.new_zeros(K).scatter_add(0, net_ids, u * exp_pos)       # [K]
        wa_max    = sum_u_exp / (sum_exp + 1e-8)                              # [K]

        # WA min: weights = exp(-u/gamma), peaks near the minimum u_i
        exp_neg       = torch.exp((-u - shift_neg) / gamma)                   # [P]
        sum_exp_neg   = u.new_zeros(K).scatter_add(0, net_ids, exp_neg)       # [K]
        sum_u_exp_neg = u.new_zeros(K).scatter_add(0, net_ids, u * exp_neg)   # [K]
        wa_min        = sum_u_exp_neg / (sum_exp_neg + 1e-8)                  # [K]
        # Note: numerator uses u_i (not -u_i) so wa_min approximates min(u_i).

        total = total + (wa_max - wa_min).sum()

    return total


# ---------------------------------------------------------------------------
# Placer class
# ---------------------------------------------------------------------------

class MSPlacer:
    """
    Analytical placer using vectorized WA HPWL + density gradient descent.

    Each iteration combines two gradient signals:
      - WL gradient  : pulls each macro toward its net centroids (attractive)
      - Density gradient: pushes macros out of overcrowded bins (repulsive)

    The density method is selected via config.toml (density_method):
      "bell"          - smooth bell-shape kernel, gradient via autograd
      "electrostatic" - ePlace Poisson FFT solver, analytic gradient

    All net/pin data is precomputed into flat tensors (vectorized) so that
    each WL forward+backward pass is a handful of scatter operations.
    """

    def __init__(self):
        cfg = _load_config()
        p = cfg.get("params", {})
        o = cfg.get("output", {})

        # ── Algorithm params ──────────────────────────────────────────────
        self.max_iters      = p.get("max_iters", 2000)
        self.seed           = p.get("seed", 42)
        # "auto" means compute from canvas size at runtime; float overrides
        self.gamma          = None if p.get("gamma", "auto") == "auto" else float(p["gamma"])
        self.max_step       = None if p.get("max_step", "auto") == "auto" else float(p["max_step"])
        self.gamma_decay    = p.get("gamma_decay", 0.98)
        self.gamma_min_frac = p.get("gamma_min_frac", 1 / 150)

        # ── Method selection ──────────────────────────────────────────────
        self.density_method       = p.get("density_method", "electrostatic")
        self.hpwl_gradient_method = p.get("hpwl_gradient_method", "wa")
        self.legalization         = p.get("legalization", "none")
        self.use_preconditioner   = p.get("use_preconditioner", True)
        self.optimizer  = p.get("optimizer", "sgd")
        # "auto" -> canvas_diag * 0.005 (same as max_step default); float overrides
        self.alpha_init = None if p.get("alpha_init", "auto") == "auto" else float(p["alpha_init"])

        # ── Lambda schedule ───────────────────────────────────────────────
        self.lambda_schedule      = p.get("lambda_schedule", "hpwl")
        self.density_weight       = p.get("density_weight", 8e-5)    # hpwl auto-init scale
        self.lambda_pcof_upper    = p.get("lambda_pcof_upper", 1.05)
        self.lambda_pcof_lower    = p.get("lambda_pcof_lower", 0.95)
        # geometric mode params (kept for compat / ablation)
        self.density_weight_init     = p.get("density_weight_init", 1e-5)
        self.density_weight_max_step = p.get("density_weight_max_step", 1.05)
        # shared
        self.warmup_iters         = p.get("warmup_iters", 50)
        # density_weight_max: hard safety cap. Without it, 1.05^1950 ≈ 10^41
        # overflows float32 → NaN in positions → crash.
        self.density_weight_max   = p.get("density_weight_max", 100.0)
        self.target_density       = p.get("target_density", 0.8)

        # ── Convergence ───────────────────────────────────────────────────
        # stop_overflow: DREAMplace Lgamma criterion — stop when overflow_ratio
        #   drops below this threshold AND WL has started rising (converged).
        #   DREAMplace default is 0.1 (10% average overflow per bin).
        self.stop_overflow    = p.get("stop_overflow", 0.1)
        # plateau_window / plateau_threshold: Lsub criterion — stop when the
        #   moving average of total loss hasn't improved by threshold over the
        #   last plateau_window iterations (DREAMplace uses window=3, thr=0.001).
        self.plateau_window    = p.get("plateau_window", 10)
        self.plateau_threshold = p.get("plateau_threshold", 0.001)
        # divergence_window: check divergence over last N iters (DREAMplace: 50).
        #   Fire if overflow is rising AND WL > 2× best WL seen so far.
        self.divergence_window = p.get("divergence_window", 50)

        # ── Initial placement ─────────────────────────────────────────────
        # "none"   = keep benchmark initial positions
        # "center" = randomize all macros near canvas center
        self.initial_placement = p.get("initial_placement", "none")
        self.initial_spread    = p.get("initial_spread", 0.01)  # fraction of canvas area

        # ── Output / logging ─────────────────────────────────────────────
        self._out = _output.OutputManager(o)

    def place(self, benchmark):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()

        _output.print_benchmark_info(benchmark, self.density_method,
                                      self.hpwl_gradient_method)

        plc = _load_plc(benchmark.name)
        if plc is None:
            print("  [warn] Could not load plc - returning initial placement", file=sys.stderr)
            return benchmark.macro_positions.clone()

        self._out.log("  Building net data...", flush=True)
        net_data = _build_net_data(benchmark, plc)
        if net_data is None:
            print("  [warn] No usable nets - returning initial placement", file=sys.stderr)
            return benchmark.macro_positions.clone()
        self._out.log(f"  {net_data['num_nets']} nets, {net_data['num_pins']} pins  "
                      f"({net_data['num_skipped']} nets skipped)")

        pos = self._gradient_place(benchmark, net_data)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:benchmark.num_macros] = pos

        # Write per-macro HPWL diagnostic (hard macros only, sorted desc)
        macro_dat = Path(self._out.frames_dir) / benchmark.name / "macros.dat"
        _output.write_macro_dat(pos, benchmark, net_data, plc, macro_dat)
        self._out.log(f"  Macro HPWL      -> {macro_dat}")

        self._out.log(f"  Total time: {time.time()-t0:.1f}s")
        return full_pos

    def _gradient_place(self, benchmark, net_data):
        """
        Gradient descent combining WL (autograd) and density (analytic) gradients.

        Each iteration:
          1. WL forward+backward  — _wa_hpwl() + autograd → wl_grad [n, 2]
          2. Density gradient     — density.compute_density_gradient() → den_grad [n, 2]
             (skipped during warmup; method selected by self.density_method)
          3. Combined gradient    — grad = wl_grad + lambda_d * den_grad
          4. Zero fixed macros
          5. Per-macro clipping: scale each gradient so movement <= max_step
          6. Gradient step: pos -= clipped_gradient
          7. Clamp positions to canvas bounds
          8. Restore fixed macros to original positions
          9. Ramp lambda_d (after warmup)
         10. Decay gamma (sharpens WA toward exact HPWL)
         11. Convergence checks (three criteria from DREAMplace NonLinearPlace.py):
             a. Lgamma: overflow < stop_overflow AND wl > prev_wl (density converged,
                WL starting to rise — optimal trade-off reached)
             b. Lsub plateau: moving-average total loss barely improving
             c. Divergence: overflow rising AND wl > 2× best wl seen
        """
        n = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        canvas_diag = max(cw, ch)

        fixed_mask = benchmark.macro_fixed[:n]
        half_w = benchmark.macro_sizes[:n, 0] / 2
        half_h = benchmark.macro_sizes[:n, 1] / 2

        gamma     = self.gamma    or canvas_diag / 8.0
        max_step  = self.max_step or canvas_diag * 0.005
        gamma_min = canvas_diag * self.gamma_min_frac

        # ── Preconditioner: p_i = macro_area_i + net_degree_i ─────────────
        # Computed once — neither macro sizes nor connectivity change during placement.
        # net_weights are currently all zeros in the loader, so we use uniform
        # weight=1 (i.e. net degree) as the connectivity term.
        if self.use_preconditioner:
            macro_area  = (benchmark.macro_sizes[:n, 0]
                           * benchmark.macro_sizes[:n, 1]).float()      # [n]
            net_degree  = torch.zeros(n, dtype=torch.float32)
            macro_ids_t = net_data["macro_ids"]                         # [P]
            is_macro_t  = net_data["is_macro"].float()                  # [P]
            net_degree.scatter_add_(0, macro_ids_t.clamp(min=0), is_macro_t)
            precond = (macro_area + net_degree).clamp(min=1.0).unsqueeze(1)  # [n, 1]
        else:
            precond = None

        pos      = benchmark.macro_positions[:n].clone().float()

        # ── Initial placement override ─────────────────────────────────────
        if self.initial_placement == "center":
            # Place all macros (including fixed) randomly within a small box
            # centered on the canvas.  spread_r = sqrt(initial_spread) * canvas
            # side gives a square with area = initial_spread * canvas_area.
            spread_r = (self.initial_spread ** 0.5)
            cx, cy   = cw / 2.0, ch / 2.0
            half_bx  = cw * spread_r / 2.0
            half_by  = ch * spread_r / 2.0
            rand_pos = torch.zeros_like(pos)
            rand_pos[:, 0] = torch.empty(n).uniform_(cx - half_bx, cx + half_bx)
            rand_pos[:, 1] = torch.empty(n).uniform_(cy - half_by, cy + half_by)
            # Clamp so each macro stays inside the canvas (accounting for half-sizes)
            rand_pos[:, 0] = rand_pos[:, 0].clamp(half_w, cw - half_w)
            rand_pos[:, 1] = rand_pos[:, 1].clamp(half_h, ch - half_h)
            pos = rand_pos

        init_pos = pos.clone()

        lambda_d = 0.0  # density weight; stays 0 during warmup, then ramps

        # Convergence tracking
        best_wl        = float("inf")  # best WL seen, for divergence detection
        best_pos       = pos.clone()   # snapshot at best WL
        prev_wl        = float("inf")  # WL from previous iter (Lgamma criterion)
        prev_overflow  = float("inf")  # overflow from previous iter
        loss_history: list[float]     = []  # for plateau detection
        overflow_history: list[float] = []  # for divergence trend
        ref_hpwl       = 1.0           # set at warmup end for hpwl lambda schedule

        # ── BB / Nesterov state ────────────────────────────────────────────
        # alpha_k: Barzilai-Borwein step size (replaces fixed max_step clip)
        # u_k / v_k: actual / look-ahead positions for Nesterov mode
        # g_prev / v_prev: previous gradient & position for BB computation
        if self.optimizer in ("bb_sgd", "nesterov"):
            alpha_k = self.alpha_init or max_step  # default same scale as old max_step
            g_prev  = None   # flattened gradient at previous eval point
            v_prev  = None   # flattened previous eval position
        if self.optimizer == "nesterov":
            u_k = pos.clone()   # actual solution
            v_k = pos.clone()   # look-ahead point
            a_k = 1.0           # Nesterov coefficient

        _alpha0 = self.alpha_init or max_step
        opt_info = (f"{self.optimizer}  alpha_init={_alpha0:.4f}  max_step={max_step:.4f}"
                    if self.optimizer != "sgd" else f"sgd  max_step={max_step:.4f}")
        self._out.log(f"  Gradient descent: {self.max_iters} iters  "
                      f"gamma0={gamma:.3f}  gamma_min={gamma_min:.4f}  "
                      f"warmup={self.warmup_iters}  stop_overflow={self.stop_overflow}  "
                      f"optimizer={opt_info}")

        self._out.setup_frames(benchmark, net_data)
        self._out.open_iter_log(benchmark)

        for t in range(self.max_iters):

            # ── Evaluate gradients at look-ahead point ────────────────────
            # Nesterov: at v_k (look-ahead); bb_sgd / sgd: at current pos.
            eval_pos = v_k if self.optimizer == "nesterov" else pos

            # ── WL gradient (via PyTorch autograd) ────────────────────────
            pos_var = eval_pos.detach().requires_grad_(True)
            wl_loss = _wa_hpwl(pos_var, net_data, gamma)
            wl_loss.backward()
            wl_grad = pos_var.grad.clone()   # [n, 2]

            # ── Density gradient (analytic, via density.py) ───────────────
            # Also compute at the warmup transition in hpwl mode so we can
            # measure gradient norms for auto-initialising lambda_d.
            need_density = (lambda_d > 0.0 or
                            (t == self.warmup_iters
                             and self.lambda_schedule == "hpwl"))
            if need_density:
                den_grad, den_energy, overflow, max_den = _density.compute_density_gradient(
                    self.density_method, eval_pos, benchmark, self.target_density)
            else:
                den_grad   = torch.zeros_like(pos)
                den_energy = 0.0
                overflow   = float("inf")
                max_den    = float("inf")

            # ── Combined gradient ─────────────────────────────────────────
            grad = wl_grad + lambda_d * den_grad
            if precond is not None:
                grad = grad / precond

            with torch.no_grad():
                # Fixed macros do not move
                grad[fixed_mask] = 0.0

                if self.optimizer in ("bb_sgd", "nesterov"):
                    # ── Barzilai-Borwein adaptive step size ───────────────
                    # Short BB: alpha = (s·y)/(y·y) where s=position diff,
                    # y=gradient diff. Falls back to Lipschitz |s|/|y| when
                    # short BB is non-positive (non-convex curvature).
                    # max_step is a hard safety cap per macro (not per scalar).
                    g_k_flat   = grad.reshape(-1)
                    eval_flat  = eval_pos.reshape(-1)

                    if g_prev is not None:
                        s_k = eval_flat - v_prev
                        y_k = g_k_flat - g_prev
                        sy  = torch.dot(s_k, y_k).item()
                        yy  = torch.dot(y_k, y_k).item()
                        ss  = torch.dot(s_k, s_k).item()
                        if sy > 0.0 and yy > 1e-20:
                            alpha_k = sy / yy       # short BB
                        elif ss > 1e-20 and yy > 1e-20:
                            alpha_k = (ss ** 0.5) / (yy ** 0.5)  # Lipschitz fallback
                        # else keep previous alpha_k

                    g_prev = g_k_flat.clone()
                    v_prev = eval_flat.clone()

                    # Scaled step with per-macro clip as safety net
                    step_vec  = alpha_k * grad
                    step_norm = step_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    step_vec  = step_vec * (max_step / step_norm).clamp(max=1.0)

                    if self.optimizer == "nesterov":
                        # ── Nesterov momentum update ──────────────────────
                        # Restart if WL is rising (landscape shifted by λ ramp
                        # or momentum overshooting). Keeps actual pos, resets
                        # look-ahead v_k and momentum coefficient a_k.
                        if prev_wl < float("inf") and wl_loss.item() > prev_wl * 1.05:
                            v_k = pos.clone()
                            a_k = 1.0

                        a_kp1 = (1.0 + (1.0 + 4.0 * a_k * a_k) ** 0.5) / 2.0
                        coef  = (a_k - 1.0) / a_kp1

                        u_kp1 = v_k.detach() - step_vec
                        u_kp1[fixed_mask] = init_pos[fixed_mask]
                        u_kp1[:, 0] = u_kp1[:, 0].clamp(min=half_w, max=cw - half_w)
                        u_kp1[:, 1] = u_kp1[:, 1].clamp(min=half_h, max=ch - half_h)

                        v_kp1 = u_kp1 + coef * (u_kp1 - u_k)
                        v_kp1[fixed_mask] = init_pos[fixed_mask]
                        v_kp1[:, 0] = v_kp1[:, 0].clamp(min=half_w, max=cw - half_w)
                        v_kp1[:, 1] = v_kp1[:, 1].clamp(min=half_h, max=ch - half_h)

                        u_k = u_kp1
                        v_k = v_kp1
                        a_k = a_kp1
                        pos = u_k
                    else:
                        # ── BB-SGD (BB step, no momentum) ─────────────────
                        pos = pos.detach() - step_vec
                        pos[:, 0] = pos[:, 0].clamp(min=half_w, max=cw - half_w)
                        pos[:, 1] = pos[:, 1].clamp(min=half_h, max=ch - half_h)
                        pos[fixed_mask] = init_pos[fixed_mask]

                    alpha = step_vec[~fixed_mask].norm(dim=1).mean().item()

                else:
                    # ── SGD with per-macro clipping ───────────────────────
                    per_macro_norm = grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    # Report gradient norm BEFORE clipping — this varies during
                    # training and is more informative than the clipped step
                    # (which is always ≈ max_step since gradients are always large).
                    alpha = per_macro_norm[~fixed_mask].mean().item()
                    scale = (max_step / per_macro_norm).clamp(max=1.0)
                    grad  = grad * scale

                    pos = pos.detach() - grad
                    pos[:, 0] = pos[:, 0].clamp(min=half_w, max=cw - half_w)
                    pos[:, 1] = pos[:, 1].clamp(min=half_h, max=ch - half_h)
                    pos[fixed_mask] = init_pos[fixed_mask]

                # NaN safety: restore from best snapshot if positions blow up
                if torch.isnan(pos).any():
                    self._out.log(f"  [NaN]       iter {t}: positions contain NaN, "
                                  f"restoring best snapshot and stopping")
                    pos = best_pos.clone()
                    if self.optimizer == "nesterov":
                        u_k = pos.clone()
                        v_k = pos.clone()
                        a_k = 1.0
                    break

            wl_val = wl_loss.item()

            # Track best WL position (for divergence recovery)
            if wl_val < best_wl:
                best_wl  = wl_val
                best_pos = pos.clone()

            # Total loss for plateau detection
            total_loss = wl_val + lambda_d * den_energy
            loss_history.append(total_loss)
            overflow_history.append(overflow)

            # ── Lambda update ─────────────────────────────────────────────
            if t >= self.warmup_iters:
                if lambda_d == 0.0:
                    if self.lambda_schedule == "hpwl":
                        # Auto-scale initial λ so density force ≈ WL force in magnitude.
                        # den_grad was computed above (need_density=True at warmup_iters).
                        wl_grad_norm  = wl_grad.norm(p=1).item()
                        den_grad_norm = den_grad.norm(p=1).item()
                        lambda_d = (self.density_weight
                                    * wl_grad_norm / (den_grad_norm + 1e-8))
                        # ref_hpwl anchors the feedback scale to current WL magnitude
                        ref_hpwl = wl_val
                    else:
                        lambda_d = self.density_weight_init
                else:
                    if self.lambda_schedule == "hpwl":
                        # DREAMplace RePlAce-style HPWL-feedback multiplier.
                        # delta_hpwl < 0: WL decreasing — ramp λ up gently.
                        # delta_hpwl > 0: WL increasing — throttle or reverse λ.
                        delta_hpwl = wl_val - prev_wl
                        if delta_hpwl < 0:
                            mu = self.lambda_pcof_upper * max(0.9999 ** t, 0.98)
                        else:
                            mu = self.lambda_pcof_upper * (
                                self.lambda_pcof_upper
                                ** (-delta_hpwl / (ref_hpwl + 1e-8))
                            )
                            mu = max(self.lambda_pcof_lower,
                                     min(self.lambda_pcof_upper, mu))
                    else:
                        mu = self.density_weight_max_step
                    lambda_d = min(lambda_d * mu, self.density_weight_max)

            # Decay gamma: sharpen WA approximation toward exact HPWL
            gamma = max(gamma * self.gamma_decay, gamma_min)

            self._out.write_iter(t, wl_val, overflow, alpha, lambda_d, gamma)

            if self._out.should_log(t, self.max_iters):
                self._out.log(f"    iter {t:4d}  wl={wl_val:.4f}  "
                              f"den={den_energy:.4f}  ovf={overflow:.4f}  "
                              f"λ={lambda_d:.2e}  gamma={gamma:.4f}")

            # ── Convergence checks (post-warmup only) ─────────────────────
            if t >= self.warmup_iters and t > 100:

                # a) Lgamma criterion (DREAMplace NonLinearPlace.py ~line 292):
                #    Density has converged (overflow below threshold) AND WL has
                #    started rising — the optimal WL/density trade-off is reached.
                if overflow < self.stop_overflow and wl_val > prev_wl:
                    self._out.log(f"  [converged] iter {t}: overflow {overflow:.4f} "
                                  f"< {self.stop_overflow} and wl rising")
                    break

                # b) Max-density criterion: every bin is already under target —
                #    no further spreading needed.
                if max_den < self.target_density:
                    self._out.log(f"  [converged] iter {t}: max_density {max_den:.4f} "
                                  f"< target {self.target_density}")
                    break

                # c) Lsub plateau criterion (DREAMplace ~line 355-372):
                #    Moving average of total loss barely improving.
                #    Only checked once density spreading has made real progress
                #    (overflow must be within 3× the stop threshold); firing
                #    earlier would terminate before lambda has ramped enough to
                #    actually spread the macros.
                w = self.plateau_window
                if (overflow < self.stop_overflow * 3
                        and len(loss_history) >= 2 * w):
                    cur_avg  = sum(loss_history[-w:])   / w
                    prev_avg = sum(loss_history[-2*w:-w]) / w
                    # DREAMplace Lsub: fire when loss is FLAT (relative change
                    # below threshold in either direction — rising loss during the
                    # density-ramp phase should NOT trigger this, only genuine
                    # convergence where loss stops moving).
                    rel_change = abs(cur_avg - prev_avg) / (prev_avg + 1e-12)
                    if rel_change < self.plateau_threshold:
                        self._out.log(f"  [plateau]   iter {t}: loss avg flat "
                                      f"{cur_avg:.4f} ≈ {prev_avg:.4f} "
                                      f"(rel={rel_change:.4f})")
                        break

                # d) Divergence detection: overflow has been RISING for the
                #    last divergence_window iters AND WL far exceeds best.
                #    We track overflow history to avoid false triggers from
                #    single-step oscillations during normal density spreading.
                dw = self.divergence_window
                if (len(overflow_history) >= dw
                        and wl_val > best_wl * 5.0):
                    # Only fire if overflow has been genuinely rising — require a
                    # relative trend of > 2% over the window, not just a tiny drift.
                    oh = overflow_history[-dw:]
                    trend = (oh[-1] - oh[0]) / (oh[0] + 1e-12)
                    if trend > 0.02:
                        self._out.log(f"  [diverged]  iter {t}: wl {wl_val:.4f} > "
                                      f"5×best {best_wl:.4f}, ovf +{trend*100:.1f}% "
                                      f"over {dw} iters; stopping (keeping current pos)")
                        # Do NOT restore best_pos — the current spread position is
                        # almost always better than the pre-warmup best snapshot.
                        break

            prev_wl       = wl_val
            prev_overflow = overflow

            self._out.save_frame(t, pos, wl_val, den_energy, overflow,
                                 lambda_d, alpha, gamma, benchmark, n)

        self._out.close()

        if self.legalization == "spiral":
            self._out.log("  Legalizing (spiral push-out)...")
            pos = _legalizer.spiral_legalize(pos, benchmark)
            self._out.save_frame(t, pos, wl_val, den_energy, overflow,
                                 lambda_d, alpha, gamma, benchmark, n)

        return pos
