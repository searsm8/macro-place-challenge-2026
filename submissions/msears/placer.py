"""
CometPlacer — Analytical WA HPWL + Density Gradient Descent

A PyTorch-based analytical placer inspired by DREAMplace (Lin et al. DAC 2019)
and ePlace (Lu et al. DAC 2015).

Placement phases:
  1. mIP  — macro initial placement (quadratic WL, center scatter, or benchmark default)
  2. mGP  — mixed global placement: WA HPWL + electrostatic density gradient descent
  2.5     — hard-macro spread: soft macros excluded from density, hard macros spread freely
  3. mLG  — macro legalization: bump or spiral push-out to remove hard-macro overlaps
  3.5     — pre-legalization rotation optimizer (greedy or SA; periodic runs during mGP)
  4. cGP  — cell global placement: hard macros fixed, soft macros re-optimized

Module structure:
  placer.py    — entry point, phase pipeline, optimizer, lambda schedule
  density.py   — density gradient (bell + electrostatic)
  legalizer.py — macro legalization (bump, spiral)
  output.py    — console logging, frame snapshots, iteration CSV
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
# Import sibling modules
# ---------------------------------------------------------------------------

def _importSibling(name):
    """Import a .py file from the same directory as this file by name."""
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_density = _importSibling("density")
_legalizer = _importSibling("legalizer")
_output = _importSibling("output")
_rotation = _importSibling("rotation")
_congestion = _importSibling("congestion")
_proxy = _importSibling("proxy")


# ---------------------------------------------------------------------------
# Phase timer
# ---------------------------------------------------------------------------

class PhaseTimer:
    """Lightweight phase-level stopwatch for placement runtime reporting."""

    def __init__(self):
        self._phases = []       # [(name, elapsed_s), ...]
        self._active = {}       # name -> perf_counter start
        self._total_start = time.perf_counter()

    def reset(self):
        self._phases = []
        self._active = {}
        self._total_start = time.perf_counter()

    def start(self, name):
        self._active[name] = time.perf_counter()

    def stop(self, name):
        t = self._active.pop(name, None)
        if t is not None:
            self._phases.append((name, time.perf_counter() - t))

    def printReport(self):
        total = time.perf_counter() - self._total_start

        def _fmt(s):
            return f"{s * 1000:.0f} ms" if s < 1.0 else f"{s:.2f} s "

        print("\n  ── Phase timing report ──────────────────────")
        print(f"  {'Phase':<24}  {'Time':>8}  {'%':>6}")
        print("  " + "─" * 44)
        for name, elapsed in self._phases:
            pct = elapsed / total * 100 if total > 0 else 0.0
            print(f"  {name:<24}  {_fmt(elapsed):>8}  {pct:>5.1f}%")
        print("  " + "─" * 44)
        print(f"  {'TOTAL':<24}  {_fmt(total):>8}  {'100.0':>5}%")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _loadConfig(path=None):
    """
    Load TOML config file. Discovery order:
      1. Explicit path argument
      2. MSPLACER_CONFIG environment variable
      3. config.toml in same directory as placer.py
    Returns empty dict if file not found.
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

def _loadPlc(name):
    """Load the PlacementCost (plc) object for a benchmark by name."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45_map = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
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

def _buildNetData(benchmark, plc):
    """
    Precompute vectorized net connectivity tensors for WA HPWL.

    Returns dict with flat parallel arrays over all P pins across K nets:
        macro_ids  LongTensor [P]     benchmark macro index (-1 = port)
        offsets    FloatTensor [P, 2] pin offset (macro) or absolute pos (port)
        net_ids    LongTensor [P]     net index per pin
        is_macro   BoolTensor [P]     True for macro pins
        num_pins   int
        num_nets   int
        num_skipped int
    """
    if plc is None:
        return None

    num_hard = benchmark.num_hard_macros
    name_to_bidx = _buildNameToBidx(plc, num_hard)
    pin_to_macro = _buildPinToMacro(plc, name_to_bidx)
    port_pos = _buildPortPositions(plc)

    return _flattenNets(plc, pin_to_macro, port_pos, net_cnt=plc.net_cnt)


def _buildNameToBidx(plc, num_hard):
    """Build mapping from plc macro name to benchmark index."""
    name_to_bidx = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx
    for i, plc_idx in enumerate(plc.soft_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = num_hard + i
    return name_to_bidx


def _buildPinToMacro(plc, name_to_bidx):
    """Build mapping from pin name to (bidx, offset_x, offset_y)."""
    pin_to_macro = {}
    for pin_plc_idx in plc.hard_macro_pin_indices + plc.soft_macro_pin_indices:
        pin = plc.modules_w_pins[pin_plc_idx]
        pin_name = pin.get_name()
        macro_name = pin_name.rsplit("/", 1)[0]
        if macro_name in name_to_bidx:
            ox, oy = pin.get_offset()
            pin_to_macro[pin_name] = (name_to_bidx[macro_name], float(ox), float(oy))
    return pin_to_macro


def _buildPortPositions(plc):
    """Build mapping from port name to absolute (x, y) position."""
    port_pos = {}
    for port_plc_idx in plc.port_indices:
        port = plc.modules_w_pins[port_plc_idx]
        px, py = port.get_pos()
        port_pos[port.get_name()] = (float(px), float(py))
    return port_pos


def _flattenNets(plc, pin_to_macro, port_pos, net_cnt=None):
    """Flatten all nets into parallel pin arrays."""
    macro_ids_list = []
    ox_list = []
    oy_list = []
    net_ids_list = []
    net_weights_list = []
    net_idx = 0
    num_skipped = 0

    for driver, sinks in plc.nets.items():
        pins_this_net = []
        for pin_name in [driver] + sinks:
            if pin_name in pin_to_macro:
                bidx, ox, oy = pin_to_macro[pin_name]
                pins_this_net.append((bidx, ox, oy))
            elif pin_name in port_pos:
                px, py = port_pos[pin_name]
                pins_this_net.append((-1, px, py))
        if len(pins_this_net) >= 2:
            driver_idx = plc.mod_name_to_indices.get(driver)
            weight = (plc.modules_w_pins[driver_idx].get_weight()
                      if driver_idx is not None else 1.0)
            for (bidx, ox, oy) in pins_this_net:
                macro_ids_list.append(bidx)
                ox_list.append(ox)
                oy_list.append(oy)
                net_ids_list.append(net_idx)
            net_weights_list.append(float(weight))
            net_idx += 1
        else:
            num_skipped += 1

    if net_idx == 0:
        return None

    macro_ids = torch.tensor(macro_ids_list, dtype=torch.long)
    offsets = torch.tensor(list(zip(ox_list, oy_list)), dtype=torch.float32)
    net_ids = torch.tensor(net_ids_list, dtype=torch.long)
    is_macro = (macro_ids >= 0)
    net_weights = torch.tensor(net_weights_list, dtype=torch.float32)

    return {
        "macro_ids": macro_ids,
        "offsets": offsets,
        "net_ids": net_ids,
        "is_macro": is_macro,
        "net_weights": net_weights,
        "num_pins": len(macro_ids_list),
        "num_nets": net_idx,
        "num_skipped": num_skipped,
        # Weighted net count from plc (Σ weight_k over all nets with sinks).
        # Used as WL normalisation denominator — matches harness exactly.
        "plc_net_cnt": float(net_cnt) if net_cnt is not None else float(net_idx),
    }


# ---------------------------------------------------------------------------
# Vectorized WA HPWL + exact HPWL
# ---------------------------------------------------------------------------

def _exactHpwl(pos, net_data):
    """Exact (non-smoothed) total HPWL: sum of per-net bounding-box spans."""
    macro_ids = net_data["macro_ids"]
    offsets   = net_data["offsets"]
    net_ids   = net_data["net_ids"]
    is_macro  = net_data["is_macro"]
    num_nets  = net_data["num_nets"]
    with torch.no_grad():
        pin_pos = _computePinPositions(pos, macro_ids, offsets, is_macro)
        max_xy = pin_pos.new_full((num_nets, 2), float("-inf"))
        min_xy = pin_pos.new_full((num_nets, 2), float("inf"))
        idx2 = net_ids.unsqueeze(1).expand(-1, 2)
        max_xy.scatter_reduce_(0, idx2, pin_pos, reduce="amax", include_self=True)
        min_xy.scatter_reduce_(0, idx2, pin_pos, reduce="amin", include_self=True)
        return float((max_xy - min_xy).sum())


def _waHpwl(pos, net_data, gamma):
    """
    Fully vectorized smooth WA HPWL using scatter_add.

    For each net k with pins at positions u_i:
        WA_max = Σ[u_i * exp(u_i/γ)] / Σ[exp(u_i/γ)]   ≈ max(u_i)
        WA_min = Σ[u_i * exp(-u_i/γ)] / Σ[exp(-u_i/γ)]  ≈ min(u_i)
        loss   = Σ_k [(WA_max - WA_min)_x + (WA_max - WA_min)_y]

    Per-net shifts prevent exp() underflow when gamma is small.
    """
    macro_ids = net_data["macro_ids"]
    offsets = net_data["offsets"]
    net_ids = net_data["net_ids"]
    is_macro = net_data["is_macro"]
    num_nets = net_data["num_nets"]

    pin_pos = _computePinPositions(pos, macro_ids, offsets, is_macro)
    total = pin_pos.new_zeros(())

    for dim in range(2):
        u = pin_pos[:, dim]
        wa_span = _waSpanForAxis(u, net_ids, num_nets, gamma)
        total = total + wa_span

    return total


def _computePinPositions(pos, macro_ids, offsets, is_macro):
    """
    Compute absolute pin positions using the port-handling trick.

    Macro pins: pos[macro_idx] * 1.0 + offset = absolute pin position
    Port pins:  pos[0] * 0.0 + absolute_pos = absolute port position
    """
    safe_ids = macro_ids.clamp(min=0)
    macro_flag = is_macro.float().unsqueeze(1)
    return pos[safe_ids] * macro_flag + offsets


def _waSpanForAxis(u, net_ids, num_nets, gamma):
    """
    Compute WA_max - WA_min summed over all nets for one axis.
    Uses per-net shifts for numerical stability.
    """
    with torch.no_grad():
        shift_pos = u.new_full((num_nets,), float("-inf"))
        shift_pos.scatter_reduce_(0, net_ids, u, reduce="amax", include_self=True)
        shift_pos = shift_pos[net_ids]

        shift_neg = u.new_full((num_nets,), float("-inf"))
        shift_neg.scatter_reduce_(0, net_ids, -u, reduce="amax", include_self=True)
        shift_neg = shift_neg[net_ids]

    # WA max
    exp_pos = torch.exp((u - shift_pos) / gamma)
    sum_exp = u.new_zeros(num_nets).scatter_add(0, net_ids, exp_pos)
    sum_u_exp = u.new_zeros(num_nets).scatter_add(0, net_ids, u * exp_pos)
    wa_max = sum_u_exp / (sum_exp + 1e-8)

    # WA min
    exp_neg = torch.exp((-u - shift_neg) / gamma)
    sum_exp_neg = u.new_zeros(num_nets).scatter_add(0, net_ids, exp_neg)
    sum_u_exp_neg = u.new_zeros(num_nets).scatter_add(0, net_ids, u * exp_neg)
    wa_min = sum_u_exp_neg / (sum_exp_neg + 1e-8)

    return (wa_max - wa_min).sum()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _asBool(val):
    """Coerce a config value to bool.

    Handles proper TOML booleans (True/False) and string representations
    ("true"/"false", "True"/"False", "1"/"0") so parameter sweeps can pass
    either form without TOML decode errors.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


# ---------------------------------------------------------------------------
# Placer class
# ---------------------------------------------------------------------------

class CometPlacer:
    """
    Analytical placer using vectorized WA HPWL + density gradient descent.

    Each iteration combines two gradient signals:
      - WL gradient  : pulls macros toward net centroids (attractive)
      - Density grad : pushes macros from overcrowded bins (repulsive)

    When config [ga] ga_enable = true, __new__ returns a GACometPlacer
    instance instead, transparently routing through the GA.
    """

    def __new__(cls, config: dict | None = None):
        if cls is CometPlacer:
            cfg = config if config is not None else _loadConfig()
            if _asBool(cfg.get("ga", {}).get("ga_enable", False)):
                ga_mod = _importSibling("ga_placer")
                return ga_mod.GACometPlacer(config=cfg)
        return super().__new__(cls)

    def __init__(self, config: dict | None = None):
        """
        Parameters
        ----------
        config : dict, optional
            Full config dict (same structure as config.toml parsed by tomllib).
            If None, the config file is auto-discovered as usual.  The GA uses
            this to inject curtailed configs without touching the filesystem.
        """
        cfg = config if config is not None else _loadConfig()
        params = cfg.get("params", {})
        output_cfg = cfg.get("output", {})
        cong_cfg = cfg.get("congestion", {})

        self._readAlgorithmParams(params)
        self._readMethodParams(params)
        self._readLambdaParams(params)
        self._readConvergenceParams(params)
        self._readPlacementParams(params)
        self._out = _output.OutputManager(output_cfg)
        self._timer = PhaseTimer()

        self.cong_rudy_enable    = _asBool(cong_cfg.get("cong_rudy_enable", False))
        self.cong_rudy_grid_size = int(cong_cfg.get("cong_rudy_grid_size", 32))
        self.lambda_cong_init            = float(cong_cfg.get("lambda_cong_init", 0.0))
        self.lambda_cong_ramp            = float(cong_cfg.get("lambda_cong_ramp", 1.05))
        self.cong_start_overflow         = float(cong_cfg.get("cong_start_overflow", 0.2))

        self.last_metrics: dict = {}

        use_gpu = cfg.get("params", {}).get("use_gpu", True)
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    def _readAlgorithmParams(self, p):
        self.mgp_enable = _asBool(p.get("mgp_enable", True))
        self.max_iters = p.get("max_iters", 2000)
        self.cgp_iters = p.get("cGP_iters", p.get("soft_place_iters", 1000))
        self.seed = p.get("seed", 42)
        self.deterministic = _asBool(p.get("deterministic", False))
        gamma_cfg = p.get("gamma", "auto")
        if gamma_cfg == "ovfw":
            self.gamma = None
            self.gamma_mode = "ovfw"
        elif gamma_cfg == "auto":
            self.gamma = None
            self.gamma_mode = "decay"
        else:
            self.gamma = float(gamma_cfg)
            self.gamma_mode = "decay"
        self.max_step_frac = float(p.get("max_step", 0.005))
        self.gamma_decay = p.get("gamma_decay", 0.98)
        self.gamma_min_frac = p.get("gamma_min_frac", 1 / 150)
        self.gamma_iters_per_update = p.get("gamma_iters_per_update", 1)

    def _readMethodParams(self, p):
        self.density_method = p.get("density_method", "electrostatic")
        self.legalization = p.get("legalization", "none")
        self.cgp_enable = _asBool(p.get("cGP_enable", p.get("soft_place", False)))
        self.cgp_position_reset = _asBool(p.get("cGP_position_reset", False))
        self.cgp_hard_macro_density_weight = float(p.get("cGP_hard_macro_density_weight", 1.0))
        self.hard_macro_density_weight = float(p.get("mGP_hard_macro_density_weight", 1.0))
        self.soft_macro_density_weight = float(p.get("mGP_soft_macro_density_weight", 1.0))
        self.hard_spread = _asBool(p.get("hard_spread", False))
        self.hard_spread_iters = p.get("hard_spread_iters", 50)
        self.halo_size = float(p.get("halo_size", 0.0))
        self.halo_legalize = float(p.get("halo_legalize", self.halo_size))
        self.hard_macro_boundary_margin = float(p.get("hard_macro_boundary_margin", 0.01))
        gs = p.get("density_grid_size", None)
        self.density_grid_rows = int(gs) if gs is not None else None
        self.density_grid_cols = int(gs) if gs is not None else None
        self.use_precond = _asBool(p.get("use_preconditioner", True))
        self.optimizer = p.get("optimizer", "sgd")
        self.alpha_init = None if p.get("alpha_init", "auto") == "auto" else float(p["alpha_init"])
        self.rotation_optimizer  = p.get("rotation_optimizer", "none")   # none/greedy/anneal/periodic
        self.rotation_passes     = p.get("rotation_passes", 1)            # greedy: full passes over all macros
        self.rotation_period     = p.get("rotation_period", 20)           # periodic: iters between rotation calls
        self.sa_T_init           = p.get("sa_T_init", 1e-4)
        self.sa_T_final          = p.get("sa_T_final", 1e-7)
        self.sa_steps_per_macro  = p.get("sa_steps_per_macro", 200)
        self.sa_mlg_outer_iters     = int(p.get("sa_mlg_outer_iters",     10))
        self.sa_mlg_steps_per_macro = int(p.get("sa_mlg_steps_per_macro", 50))
        self.sa_mlg_beta            = float(p.get("sa_mlg_beta",          1.5))
        _cand_map = {
            "NS":       (0, 2),
            "no-swap":  (0, 1, 2, 3),
            "all":      (0, 1, 2, 3, 4, 5, 6, 7),
        }
        self.rotation_candidates  = _cand_map[p.get("rotation_candidates", "all")]
        self.n_placement_passes   = p.get("n_placement_passes", 1)

    def _readLambdaParams(self, p):
        self.lambda_den_schedule = p.get("lambda_den_schedule", "hpwl")
        self.lambda_den_init = p.get("lambda_den_init", 8e-4)
        self.lambda_den_pcof_upper = p.get("lambda_den_pcof_upper", 1.05)
        self.lambda_den_pcof_lower = p.get("lambda_den_pcof_lower", 0.95)
        self.lambda_den_weight_init = p.get("lambda_den_weight_init", 1e-5)
        self.lambda_den_max_step = p.get("lambda_den_max_step", 1.04)
        self.warmup_iters = p.get("warmup_iters", 20)
        self.lambda_den_max = p.get("lambda_den_max", 5000.0)
        self.target_density = p.get("target_density", 0.5)
        # Hard-macro density boost: multiplies density gradient for hard macros
        # only, starts at lambda_hm_init and decays toward 1.0 each iteration.
        self.lambda_hm_init = p.get("lambda_hm_init", 3.0)
        self.lambda_hm_decay = p.get("lambda_hm_decay", 0.995)
        self.lambda_den_iters_per_update = p.get("lambda_den_iters_per_update", 1)

    def _readConvergenceParams(self, p):
        self.stop_overflow = p.get("stop_overflow", 0.1)
        self.plateau_window = p.get("plateau_window", 10)
        self.plateau_thresh = p.get("plateau_threshold", 0.001)
        self.divergence_window = p.get("divergence_window", 50)
        self.conv_lgamma     = _asBool(p.get("conv_lgamma",      True))
        self.conv_max_den    = _asBool(p.get("conv_max_density",  True))
        self.conv_plateau    = _asBool(p.get("conv_plateau",      True))
        self.conv_lambda_max = _asBool(p.get("conv_lambda_max",   True))
        self.conv_divergence = _asBool(p.get("conv_divergence",   True))
        self.convergence_countdown = p.get("convergence_countdown", 30)

    def _readPlacementParams(self, p):
        self.init_placement = p.get("initial_placement", "none")
        self.init_spread = p.get("center_init_spread", 0.01)
        self.quad_b2b_iters = p.get("quad_b2b_iters", 3)
        self.quad_net_size_threshold = p.get("quad_net_size_threshold", 50)
        self.quad_anchor_fraction   = p.get("quad_anchor_fraction", 0.0)
        self.quad_scatter_n         = p.get("quad_scatter_n", 0)
        self.quad_scatter_fraction  = p.get("quad_scatter_fraction", 0.0)
        self.scatter_lock_mult      = p.get("quad_scatter_lock_mult", 0.0)
        self.quad_scatter_section_count    = int(p.get("quad_scatter_section_count", 1))
        self.quad_scatter_runs_per_section = int(p.get("quad_scatter_runs_per_section", 1))
        # Runtime section context — updated per run in multi-section mode
        self._section_idx   = 0
        self._section_count = 1

    def placeWithData(self, benchmark, net_data: dict):
        """
        Run placement with a pre-built *net_data* dict.

        Used by the GA to skip plc reloading and net_data construction for
        each fitness evaluation.  Output logging and frame recording are
        suppressed according to the placer's own output config (set
        ``quiet=True`` and ``record_frames=False`` in the injected config for
        silent GA evaluation runs).

        Sets ``self.last_metrics = {"wl": float, "overflow": float}`` after
        the run so the GA can read the fitness signal.

        Returns
        -------
        full_pos : FloatTensor [num_total_macros, 2]
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        pos, _ = self._runPlacementPipeline(benchmark, net_data)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:benchmark.num_macros] = pos
        return full_pos

    # -------------------------------------------------------------------
    # Rotation optimizer
    # -------------------------------------------------------------------

    # Our internal orientation indices 4-7 (E, FE, W, FW) have opposite rotation
    # direction to the plc's same-named orientations.  The correct plc name for
    # each of our indices is:  N→N  FN→FN  S→S  FS→FS  E→W  FE→FW  W→E  FW→FE
    _OUR_ORI_TO_PLC = ["N", "FN", "S", "FS", "W", "FW", "E", "FE"]

    @staticmethod
    def _applyOrientationSizes(old_genes, new_genes, benchmark):
        """
        Swap macro_sizes w↔h for any macro that crosses the E/W boundary.
        Orientations 4-7 (E/FE/W/FW) physically transpose the footprint; 0-3 do not.
        Only swaps when the boundary is crossed (old < 4 and new >= 4, or vice versa),
        so calling this repeatedly is safe — no double-swaps.
        Returns the number of macros whose sizes were swapped.
        """
        n_swapped = 0
        for m in range(benchmark.num_hard_macros):
            was_ew = int(old_genes[m]) >= 4
            now_ew = int(new_genes[m]) >= 4
            if was_ew != now_ew:
                w = benchmark.macro_sizes[m, 0].clone()
                benchmark.macro_sizes[m, 0] = benchmark.macro_sizes[m, 1]
                benchmark.macro_sizes[m, 1] = w
                n_swapped += 1
        return n_swapped

    @staticmethod
    def _buildRotatedNetData(genes, net_data_base):
        """
        Return a new net_data dict whose offsets reflect the given gene orientations,
        applied from the all-N baseline in net_data_base.  Used to give Pass 2 the
        correct starting pin positions without re-running _buildNetData from the plc.
        """
        offsets   = net_data_base["offsets"].clone()
        macro_ids = net_data_base["macro_ids"]
        for m, gene in enumerate(genes.tolist()):
            if gene == 0:
                continue
            m_mask = macro_ids == m
            if not m_mask.any():
                continue
            mat = _rotation._ORI_MATRICES[gene]
            dx = offsets[m_mask, 0].clone()
            dy = offsets[m_mask, 1].clone()
            offsets[m_mask, 0] = mat[0] * dx + mat[1] * dy
            offsets[m_mask, 1] = mat[2] * dx + mat[3] * dy
        return {**net_data_base, "offsets": offsets}

    def _runRotationOptimizer(self, pos, net_data, net_data_base, genes, benchmark, gamma, plc=None):
        """
        Call the appropriate rotation function (greedy or SA), log results,
        then sync the chosen orientations back into the plc object so the
        harness evaluator sees the rotated pin positions.

        All 8 orientations (N, FN, S, FS, E, FE, W, FW) are safe to use:
        - plc.update_macro_orientation only rotates pin offsets, never swaps
          macro dimensions, so legalized positions remain overlap-free.
        - The harness validates overlaps against original macro_sizes and has
          no concept of orientation — it cannot flag rotated macros as illegal.
        - _OUR_ORI_TO_PLC maps our rotation-matrix indices to the plc names,
          correcting for the opposite direction convention of E/W.
        """
        candidates = self.rotation_candidates
        # net_data_base is always CPU; rotation ops must run on a consistent device.
        # Work on CPU copies of pos and work_offs, then sync work_offs back.
        pos_cpu       = pos.detach().cpu()
        work_offs_dev = net_data["offsets"]                  # may be on CUDA
        work_offs_cpu = work_offs_dev.cpu().clone()
        old_genes     = genes.clone()

        mode = self.rotation_optimizer
        if mode in ("greedy", "periodic"):
            n = _rotation.greedy_rotate(
                pos_cpu, work_offs_cpu, net_data_base,
                genes, benchmark, gamma=gamma,
                n_passes=self.rotation_passes,
                candidates=candidates,
            )
        elif mode == "anneal":
            n_steps = self.sa_steps_per_macro * benchmark.num_hard_macros
            n = _rotation.sa_rotate(
                pos_cpu, work_offs_cpu, net_data_base,
                genes, benchmark, gamma=gamma,
                n_steps=n_steps,
                T_init=self.sa_T_init,
                T_final=self.sa_T_final,
                candidates=candidates,
            )

        # Sync rotated offsets back to the original device tensor.
        work_offs_dev.copy_(work_offs_cpu.to(work_offs_dev.device))

        # Swap macro_sizes w↔h for macros that crossed the E/W boundary.
        n_swapped = self._applyOrientationSizes(old_genes, genes, benchmark)
        self._out.log(f"  Rotation (pre-legalization): {n} macro(s) rotated, "
                      f"{n_swapped} size-swapped")

        # Sync all non-N orientations to plc so harness sees rotated pin positions.
        # Assumes plc is in N orientation when this is called (true because _loadPlc()
        # returns a freshly-loaded plc, and update_macro_orientation is cumulative).
        # update_macro_orientation("N") is a no-op in the plc, so we skip N genes.
        if plc is not None:
            for m in range(benchmark.num_hard_macros):
                gene = int(genes[m])
                if gene != 0:
                    plc_idx = plc.hard_macro_indices[m]
                    plc.update_macro_orientation(plc_idx, self._OUR_ORI_TO_PLC[gene])

    # -------------------------------------------------------------------
    # Top-level entry point
    # -------------------------------------------------------------------

    def place(self, benchmark):
        total_runs = self.quad_scatter_section_count * self.quad_scatter_runs_per_section

        torch.manual_seed(self.seed)
        # Seed the CUDA RNG explicitly — torch.manual_seed() alone may not cover it
        # on all PyTorch versions.
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.deterministic:
            # Must be set before CUDA initializes; harmless if already set.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        start_time = time.time()
        self._timer.reset()

        _output.printBenchmarkInfo(benchmark, self.density_method)

        if total_runs > 1:
            return self._multiSectionPlace(benchmark, total_runs, start_time)

        self._timer.start("setup")
        plc = _loadPlc(benchmark.name)
        if plc is None:
            print("  [warn] Could not load plc - returning initial placement",
                  file=sys.stderr, flush=True)
            return benchmark.macro_positions.clone()

        self._out.log("  Building net data...", flush=True)
        net_data = _buildNetData(benchmark, plc)
        if net_data is None:
            print("  [warn] No usable nets - returning initial placement",
                  file=sys.stderr, flush=True)
            return benchmark.macro_positions.clone()
        self._out.log(f"  {net_data['num_nets']} nets, {net_data['num_pins']} pins  "
                      f"({net_data['num_skipped']} nets skipped)")
        self._timer.stop("setup")

        # Base net_data (all-N offsets) and gene state — used by rotation optimizer.
        net_data_base   = net_data
        current_genes   = torch.zeros(benchmark.num_hard_macros, dtype=torch.int8)

        genes = current_genes
        nd    = net_data
        for pass_idx in range(self.n_placement_passes):
            if pass_idx > 0:
                print(f"\n  ── Placement pass {pass_idx + 1} / {self.n_placement_passes} "
                      f"(locked orientations from pass {pass_idx}) ──")
                # Rebuild net_data with pass (pass_idx-1)'s orientations applied to N-baseline.
                nd = self._buildRotatedNetData(genes, net_data_base)
                # Reload plc: update_macro_orientation is cumulative and "N" is a no-op,
                # so there is no way to reset pin offsets without reloading from disk.
                plc = _loadPlc(benchmark.name)
            pos, genes = self._runPlacementPipeline(
                benchmark, nd, net_data_base, genes, plc,
                run_rotation=(pass_idx == 0),
            )

        # Sync final orientations to plc. On pass 1 this was done inside
        # _runRotationOptimizer; on locked passes rotation is skipped so the
        # freshly-reloaded plc (N orientation) needs an explicit sync here.
        if plc is not None:
            for m in range(benchmark.num_hard_macros):
                gene = int(genes[m])
                if gene != 0:
                    plc.update_macro_orientation(
                        plc.hard_macro_indices[m], self._OUR_ORI_TO_PLC[gene]
                    )

        full_pos = benchmark.macro_positions.clone()
        full_pos[:benchmark.num_macros] = pos

        macro_dat = Path(self._out.frames_dir).parent / "data" / benchmark.name / "macros.dat"
        _output.writeMacroDat(pos, benchmark, net_data, plc, macro_dat)
        self._out.log(f"  Macro HPWL      -> {macro_dat}")

        from macro_place.objective import compute_proxy_cost
        costs = compute_proxy_cost(pos, benchmark, plc)
        self._out.log(
            f"  Proxy score     : {costs['proxy_cost']:.4f}  "
            f"(wl={costs['wirelength_cost']:.4f}  "
            f"den={costs['density_cost']:.4f}  "
            f"cong={costs['congestion_cost']:.4f}  "
            f"overlaps={costs['overlap_count']})"
        )
        self._out.saveProxyScore(costs)
        _proxy.validate(pos, net_data, benchmark, plc)

        if self.cong_rudy_enable:
            gs = self.cong_rudy_grid_size
            net_data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in net_data.items()}
            rudy_map = _congestion.compute_rudy_map(
                pos.cpu(), net_data_cpu,
                benchmark.canvas_width, benchmark.canvas_height,
                gs, gs,
            )
            stats = _congestion.rudy_stats(rudy_map)
            self._out.log(
                f"  RUDY congestion : mean={stats['rudy_mean']:.4f}  "
                f"p99={stats['rudy_p99']:.4f}  max={stats['rudy_max']:.4f}"
                f"  (grid={gs}×{gs})"
            )

        self._out.log(f"  Total time: {time.time()-start_time:.1f}s")
        self._timer.printReport()
        return full_pos

    # -------------------------------------------------------------------
    # Multi-section placement loop
    # -------------------------------------------------------------------

    def _multiSectionPlace(self, benchmark, total_runs, start_time):
        """
        Run the full placement pipeline total_runs times, sweeping the largest
        scatter macro across canvas sections and varying the RNG seed.

        Seeds are derived deterministically from the master seed:
            run k uses seed = master_seed + k

        Returns the full_pos of the run with the best proxy cost (0-overlap
        runs preferred over runs with overlaps).
        """
        from macro_place.objective import compute_proxy_cost

        master_seed = self.seed
        n_sec = self.quad_scatter_section_count

        saved_out = self._out
        quiet_out = _output.OutputManager({
            "quiet": True,
            "record_frames": False,
            "frames_dir": saved_out.frames_dir,
        })

        initial_macro_sizes = benchmark.macro_sizes.clone()

        best_proxy       = float("inf")
        best_full_pos    = None
        best_macro_sizes = None
        best_costs       = None
        best_has_valid   = False   # True when best run has 0 overlaps

        saved_out.log(
            f"  [multi-section] {total_runs} total runs  "
            f"sections={n_sec}  runs_per_section={self.quad_scatter_runs_per_section}"
        )

        time_limit_s  = 55 * 60   # stop scheduling new runs past this elapsed time
        run_durations = []        # wall-clock seconds for each completed run

        for run_idx in range(total_runs):
            # Time-budget check: if elapsed + projected next run would exceed limit, stop.
            elapsed = time.time() - start_time
            if run_durations:
                avg_run = sum(run_durations) / len(run_durations)
                if elapsed + avg_run > time_limit_s:
                    saved_out.log(
                        f"  [multi-section] time budget: elapsed={elapsed/60:.1f}m  "
                        f"avg_run={avg_run/60:.1f}m  projected={( elapsed+avg_run)/60:.1f}m > 55m  "
                        f"stopping after {run_idx}/{total_runs} runs"
                    )
                    break

            section_idx        = run_idx // self.quad_scatter_runs_per_section
            within_section_idx = run_idx  % self.quad_scatter_runs_per_section
            # Seed layout: master + section + within * n_sections
            # Guarantees that run 0 of each section is stable across runs_per_section
            # values, so higher runs_per_section strictly extends the search.
            run_seed = master_seed + section_idx + within_section_idx * n_sec

            run_start = time.time()

            # Restore benchmark state modified by previous run's orientation swaps
            if run_idx > 0:
                benchmark.macro_sizes.copy_(initial_macro_sizes)

            # Set per-run context (used by _initialPlacement → quadratic_init)
            self._section_idx   = section_idx
            self._section_count = n_sec
            self.seed           = run_seed
            self._out           = quiet_out

            torch.manual_seed(run_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(run_seed)
            np.random.seed(run_seed)

            # Load plc + net_data
            plc = _loadPlc(benchmark.name)
            if plc is None:
                saved_out.log(f"  [run {run_idx+1}/{total_runs}] plc load failed, skipping")
                continue
            net_data = _buildNetData(benchmark, plc)
            if net_data is None:
                saved_out.log(f"  [run {run_idx+1}/{total_runs}] net_data build failed, skipping")
                continue

            net_data_base = net_data
            genes = torch.zeros(benchmark.num_hard_macros, dtype=torch.int8)
            nd    = net_data

            for pass_idx in range(self.n_placement_passes):
                if pass_idx > 0:
                    nd  = self._buildRotatedNetData(genes, net_data_base)
                    plc = _loadPlc(benchmark.name)
                pos, genes = self._runPlacementPipeline(
                    benchmark, nd, net_data_base, genes, plc,
                    run_rotation=(pass_idx == 0),
                )

            # Sync final orientations to plc
            if plc is not None:
                for m in range(benchmark.num_hard_macros):
                    gene = int(genes[m])
                    if gene != 0:
                        plc.update_macro_orientation(
                            plc.hard_macro_indices[m], self._OUR_ORI_TO_PLC[gene]
                        )

            full_pos = benchmark.macro_positions.clone()
            full_pos[:benchmark.num_macros] = pos

            costs  = compute_proxy_cost(pos, benchmark, plc)
            proxy  = costs["proxy_cost"]
            n_ovlp = costs["overlap_count"]
            is_valid = (n_ovlp == 0)

            run_durations.append(time.time() - run_start)

            saved_out.log(
                f"  [run {run_idx+1:2d}/{total_runs}]  "
                f"sec={section_idx}  seed={run_seed}  "
                f"proxy={proxy:.4f}  "
                f"wl={costs['wirelength_cost']:.4f}  "
                f"den={costs['density_cost']:.4f}  "
                f"cong={costs['congestion_cost']:.4f}  "
                f"overlaps={n_ovlp}  "
                f"({run_durations[-1]/60:.1f}m)"
            )

            # Prefer 0-overlap; among equal validity, take lower proxy
            if (is_valid and not best_has_valid) or (
                is_valid == best_has_valid and proxy < best_proxy
            ):
                best_proxy       = proxy
                best_full_pos    = full_pos.clone()
                best_macro_sizes = benchmark.macro_sizes.clone()
                best_costs       = costs
                best_has_valid   = is_valid

        # Restore placer state
        self._out           = saved_out
        self.seed           = master_seed
        self._section_idx   = 0
        self._section_count = 1

        if best_full_pos is None:
            return benchmark.macro_positions.clone()

        benchmark.macro_sizes.copy_(best_macro_sizes)

        saved_out.log(
            f"\n  [multi-section] best: proxy={best_costs['proxy_cost']:.4f}  "
            f"(wl={best_costs['wirelength_cost']:.4f}  "
            f"den={best_costs['density_cost']:.4f}  "
            f"cong={best_costs['congestion_cost']:.4f}  "
            f"overlaps={best_costs['overlap_count']})"
        )
        saved_out.saveProxyScore(best_costs)
        saved_out.log(f"  Total time: {time.time()-start_time:.1f}s")

        return best_full_pos

    # -------------------------------------------------------------------
    # Gradient descent loop
    # -------------------------------------------------------------------

    def _runPlacementPipeline(self, benchmark, net_data, net_data_base=None, current_genes=None, plc=None, run_rotation=True):
        """Orchestrates all placement phases: mGP → hard spread → mLG → [rotation] → cGP."""
        net_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in net_data.items()}

        state = self._initState(benchmark, net_data)
        rot_genes = (current_genes.clone()
                     if current_genes is not None
                     else torch.zeros(benchmark.num_hard_macros, dtype=torch.int8))

        # Phase 2: mGP
        if self.mgp_enable:
            self._out.banner("Phase 2: mGP")
            self._timer.start("mGP")
            last_t = self._mixedGlobalPlacement(state, benchmark, net_data, rot_genes, net_data_base, run_rotation)
            self._timer.stop("mGP")
        else:
            self._out.log("  [mgp_enable=false] skipping mGP")
            last_t = 0
            # Compute WL from initial placement so last_metrics has a real signal.
            nd_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in net_data.items()}
            state["prev_wl"] = float(_exactHpwl(state["pos"].detach().cpu(), nd_cpu))
            state["overflow_history"] = [1.0]  # no density info available

        # Phase 2.5: hard-macro spread
        self._out.log(f"  hard_spread={self.hard_spread}  "
                      f"hard_spread_iters={self.hard_spread_iters}")
        if self.hard_spread and benchmark.num_soft_macros > 0:
            self._out.banner("Phase 2.5: hard spread")
            self._timer.start("hard spread")
            last_t = self._hardMacroSpread(state, benchmark, net_data, frame_offset=last_t + 1)
            self._timer.stop("hard spread")

        # Phase 3.5: pre-legalization rotation optimizer (skipped on locked passes)
        if run_rotation and self.rotation_optimizer in ("greedy", "anneal") and net_data_base is not None:
            self._out.banner("Phase 3.5: rotation")
            self._timer.start("rotation")
            self._runRotationOptimizer(
                state["pos"], net_data, net_data_base, rot_genes, benchmark,
                state["gamma"], plc
            )
            self._timer.stop("rotation")
            self._out.banner("Phase 3.5: rotation  done")

        # --- snapshot after phase 2 ---
        nh = benchmark.num_hard_macros
        sizes_cpu = benchmark.macro_sizes.cpu()
        net_data_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in net_data.items()}
        pos_p2 = state["pos"].detach().cpu()
        ovfw_p2 = state["overflow_history"][-1] if state["overflow_history"] else None
        snapshots = [{"phase": "2: mGP", "hpwl": _exactHpwl(pos_p2, net_data_cpu), "overflow": ovfw_p2, "hard_overlaps": _legalizer._countOverlaps(pos_p2, nh, sizes_cpu), "soft_disp": None}]

        # Phase 3: mLG — reset best_pos to legalized result so cGP tracks from clean baseline
        self._out.banner("Phase 3: mLG")
        self._timer.start("mLG")
        self._macroLegalization(state, last_t, benchmark, net_data)
        self._timer.stop("mLG")
        self._out.banner("Phase 3: mLG  done")
        state["best_pos"] = state["pos"].clone()

        # --- snapshot after phase 3 ---
        pos_p3 = state["pos"].detach().cpu()
        soft_disp_p3 = float((pos_p3[nh:] - pos_p2[nh:]).norm(dim=1).mean()) if benchmark.num_soft_macros > 0 else None
        snapshots.append({"phase": "3: mLG", "hpwl": _exactHpwl(pos_p3, net_data_cpu), "overflow": None, "hard_overlaps": _legalizer._countOverlaps(pos_p3, nh, sizes_cpu), "soft_disp": soft_disp_p3})

        # Phase 4: cGP
        if self.cgp_enable and benchmark.num_soft_macros > 0:
            self._out.banner("Phase 4: cGP")
            self._timer.start("cGP")
            state["pos"] = self._cellGlobalPlacement(state["pos"], benchmark, net_data,
                                                     frame_offset=last_t + 1)
            self._timer.stop("cGP")

            # --- snapshot after phase 4 ---
            pos_p4 = state["pos"].detach().cpu()
            ovfw_p4 = state["overflow_history"][-1] if state["overflow_history"] else None
            self._out.saveCgpFrame(last_t, state["pos"], state["prev_wl"],
                                   ovfw_p4, state["gamma"], benchmark, state["num_macros"])
            soft_disp_p4 = float((pos_p4[nh:] - pos_p2[nh:]).norm(dim=1).mean())
            snapshots.append({"phase": "4: cGP", "hpwl": _exactHpwl(pos_p4, net_data_cpu), "overflow": ovfw_p4, "hard_overlaps": _legalizer._countOverlaps(pos_p4, nh, sizes_cpu), "soft_disp": soft_disp_p4})

        self._out.writeRunSummary(benchmark, snapshots)
        self._out.close()

        self.last_metrics = {
            "wl": float(state["prev_wl"]),
            "overflow": float(
                state["overflow_history"][-1] if state["overflow_history"] else 1.0
            ),
        }
        return state["pos"].cpu(), rot_genes

    def _mixedGlobalPlacement(self, state, benchmark, net_data, rot_genes, net_data_base=None, run_rotation=True):
        """Phase 2 (mGP): WA HPWL + electrostatic density gradient descent main loop."""
        for t in range(self.max_iters):
            should_stop = self._runOneIteration(t, state, benchmark, net_data)
            if should_stop:
                break

            if (run_rotation
                    and self.rotation_optimizer == "periodic"
                    and net_data_base is not None
                    and t > self.warmup_iters
                    and t % self.rotation_period == 0):
                pos_cpu       = state["pos"].detach().cpu()
                work_offs_dev = net_data["offsets"]
                work_offs_cpu = work_offs_dev.cpu().clone()
                old_genes     = rot_genes.clone()
                n = _rotation.greedy_rotate(
                    pos_cpu, work_offs_cpu, net_data_base,
                    rot_genes, benchmark, gamma=state["gamma"],
                    candidates=self.rotation_candidates,
                )
                work_offs_dev.copy_(work_offs_cpu.to(work_offs_dev.device))
                n_swapped = self._applyOrientationSizes(old_genes, rot_genes, benchmark)
                print(f"    [rotation t={t}] {n} macro(s) rotated, {n_swapped} size-swapped")
        self._out.log(f"  [mGP] done  iters={t + 1}")
        self._out.banner(f"Phase 2: mGP  done (iters={t + 1})")
        return t

    def _macroLegalization(self, state, last_t, benchmark, net_data):
        """Phase 3 (mLG): remove hard-macro overlaps via bump, spiral, or SA."""
        if self.legalization == "bump":
            self._out.log("  Legalizing (pairwise bump)...")
            state["pos"] = _legalizer.bumpLegalize(
                state["pos"].cpu(), benchmark, halo_size=self.halo_legalize,
                verbose=self._out.legalization_details,
                quiet=self._out.quiet,
            ).to(self.device)
            last_ovfw = state["overflow_history"][-1] if state["overflow_history"] else 0.0
            self._out.saveLegalFrame(last_t, state["pos"], state["prev_wl"],
                                     last_ovfw, state["gamma"], benchmark, state["num_macros"])
        elif self.legalization == "spiral":
            self._out.log("  Legalizing (spiral push-out)...")
            state["pos"] = _legalizer.spiralLegalize(
                state["pos"].cpu(), benchmark,
                verbose=self._out.legalization_details,
                quiet=self._out.quiet,
            ).to(self.device)
            last_ovfw = state["overflow_history"][-1] if state["overflow_history"] else 0.0
            self._out.saveLegalFrame(last_t, state["pos"], state["prev_wl"],
                                     last_ovfw, state["gamma"], benchmark, state["num_macros"])
        elif self.legalization == "sa":
            self._out.log("  Legalizing (SA)...")
            nd_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in net_data.items()}
            state["pos"] = _legalizer.saLegalize(
                state["pos"].cpu(), benchmark, nd_cpu,
                outer_iters=self.sa_mlg_outer_iters,
                steps_per_macro=self.sa_mlg_steps_per_macro,
                beta=self.sa_mlg_beta,
                quiet=self._out.quiet,
            ).to(self.device)
            last_ovfw = state["overflow_history"][-1] if state["overflow_history"] else 0.0
            self._out.saveLegalFrame(last_t, state["pos"], state["prev_wl"],
                                     last_ovfw, state["gamma"], benchmark, state["num_macros"])

    def _hardMacroSpread(self, state, benchmark, net_data, frame_offset=0):
        """
        Phase 2.5: let hard macros spread before legalization.

        Soft macros are excluded from the density map so they no longer add
        "mass" to any bin, but all macros still feel the electric field
        generated by the hard-macro-only density. This gives hard macros room
        to separate without the soft-macro density peaks competing for space.

        Uses plain SGD (no BB/Nesterov momentum) to avoid stale state from
        the main loop.  Lambda and gamma continue from where phase 2 left off.
        Returns the last frame index used, so the legalization frame is numbered
        correctly.
        """
        density_mask = state["hard_mask"]   # only hard macros feed the density map

        # Reset lambda to the base density_weight — by convergence lambda_d can
        # be 100–5000×, which would make the phase 2.5 force overwhelming.
        prev_lambda = state["lambda_d"]
        state["lambda_d"] = self.lambda_den_init

        self._out.log(
            f"  Phase 2.5 (hard spread): {self.hard_spread_iters} iters — "
            f"soft macros excluded from density map  "
            f"lambda reset {prev_lambda:.2e} -> {self.lambda_den_init:.2e}")

        last_t = frame_offset
        for s in range(self.hard_spread_iters):
            t = frame_offset + s
            last_t = t

            wl_grad, wl_val = self._computeWlGradient(
                state["pos"], net_data, state["gamma"])
            den_grad, den_energy, overflow, _ = _density.computeDensityGradient(
                self.density_method, state["pos"], benchmark,
                self.target_density, density_mask)

            grad = self._combineGradients(wl_grad, den_grad, state)

            with torch.no_grad():
                grad[state["fixed_mask"]] = 0.0
                self._stepSgd(grad, state)

                if torch.isnan(state["pos"]).any():
                    self._out.log(f"  [NaN] phase 2.5 iter {s}: restoring best")
                    state["pos"] = state["best_pos"].clone()
                    break

            self._trackBestWl(wl_val, overflow, state)
            if s % self.gamma_iters_per_update == 0:
                self._stepGamma(state, overflow)

            if self._out.shouldLog(s, self.hard_spread_iters):
                self._out.log(
                    f"    [2.5] iter {s:4d}  wl={wl_val:.4f}  "
                    f"den={den_energy:.4f}  ovf={overflow:.4f}  "
                    f"\u03bb={state['lambda_d']:.2e}")

            self._out.saveFrame(t, state["pos"], wl_val, den_energy, overflow,
                                state["lambda_d"], 0.0, state["gamma"],
                                benchmark, state["num_macros"], phase="2.5: mGP")

            state["prev_wl"] = wl_val
            state["prev_overflow"] = overflow

        self._out.log(f"  [2.5] done  iters={self.hard_spread_iters}")
        self._out.banner(f"Phase 2.5: hard spread  done (iters={self.hard_spread_iters})")
        return last_t

    def _cellGlobalPlacement(self, pos, benchmark, net_data, frame_offset=0):
        """
        Phase 4 (cGP): soft-macro global placement after hard-macro legalization.

        Hard macros are locked at their legalized positions. Soft macros are
        re-optimized with WL + density gradient descent to fill the whitespace
        left by mLG, reducing congestion and wirelength.

        Uses SGD (fresh start — no stale BB/Nesterov momentum from phase 2).
        Frame saving is skipped; logs to the same console output.
        """
        num_macros = benchmark.num_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        canvas_diag = max(canvas_w, canvas_h)

        # Lock all hard macros (originally-fixed ones already set; add the rest)
        fixed_mask = benchmark.macro_fixed.clone().to(self.device)
        fixed_mask[:benchmark.num_hard_macros] = True

        half_w = benchmark.macro_sizes[:num_macros, 0].to(self.device) / 2
        half_h = benchmark.macro_sizes[:num_macros, 1].to(self.device) / 2
        # Hard macros are fixed in cGP; no boundary margin needed.
        cgp_clamp_lo_x = half_w
        cgp_clamp_hi_x = canvas_w - half_w
        cgp_clamp_lo_y = half_h
        cgp_clamp_hi_y = canvas_h - half_h

        # Restart gamma and step size fresh for this phase
        gamma = canvas_diag / 8.0
        gamma_min = canvas_diag * self.gamma_min_frac
        max_step = canvas_diag * self.max_step_frac

        state = {
            "num_macros": num_macros,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "gamma": gamma,
            "gamma_base": gamma / 10.0,
            "max_step": max_step,
            "gamma_min": gamma_min,
            "fixed_mask": fixed_mask,
            "half_w": half_w,
            "half_h": half_h,
            "clamp_lo_x": cgp_clamp_lo_x,
            "clamp_hi_x": cgp_clamp_hi_x,
            "clamp_lo_y": cgp_clamp_lo_y,
            "clamp_hi_y": cgp_clamp_hi_y,
            "precond": self._buildPreconditioner(benchmark, net_data, num_macros),
            "pos": pos.clone(),
            "init_pos": pos.clone(),  # _clampPositions restores fixed macros to this
            "lambda_d": 0.0,
            "lambda_cong_eff": 0.0,
            "lambda_cong_target": 0.0,
            "lambda_hm": 1.0,         # no hard-macro boost; they're fixed anyway
            "hard_mask": benchmark.get_hard_macro_mask().to(self.device),
            "hard_macro_weight": self.cgp_hard_macro_density_weight,
            "best_wl": float("inf"),
            "best_pos": pos.clone(),
            "prev_wl": float("inf"),
            "prev_overflow": float("inf"),
            "loss_history": [],
            "overflow_history": [],
            "ref_hpwl": 1.0,
            "overflow_ema": float("inf"),
            "stop_reason": "",
            "conv_life": None,
            "macro_locks": [],
            "active_lock_mask": None,
            "scatter_lock_mask": torch.zeros(num_macros, dtype=torch.bool, device=self.device),
        }

        # Optionally reset soft macros to canvas center before optimizing.
        # Hard macro positions are locked via fixed_mask and restored each iter.
        if self.cgp_position_reset:
            num_hard = benchmark.num_hard_macros
            with torch.no_grad():
                state["pos"][num_hard:, 0] = canvas_w / 2.0
                state["pos"][num_hard:, 1] = canvas_h / 2.0
            state["best_pos"] = state["pos"].clone()

        self._out.log(
            f"  Phase 4 (cGP): {benchmark.num_soft_macros} soft macros free  "
            f"max_iters={self.cgp_iters}  warmup={self.warmup_iters}  "
            f"gamma0={gamma:.3f}  pos_reset={self.cgp_position_reset}")

        for t in range(self.cgp_iters):
            eval_pos = state["pos"]

            wl_grad, wl_val = self._computeWlGradient(eval_pos, net_data, state["gamma"])
            den_grad, den_energy, overflow, max_den = self._computeDenGradient(
                t, eval_pos, benchmark, state)
            cong_grad = self._computeCongGradient(
                t, eval_pos, net_data, benchmark, state, wl_grad, overflow)

            grad = self._combineGradients(wl_grad, den_grad, state, cong_grad)

            with torch.no_grad():
                grad[fixed_mask] = 0.0
                alpha = self._stepSgd(grad, state)

            self._trackBestWl(wl_val, overflow, state)
            if t % self.lambda_den_iters_per_update == 0:
                self._updateLambda(t, wl_val, wl_grad, den_grad, den_energy, state,
                                   overflow=overflow)
            if t % self.gamma_iters_per_update == 0:
                self._stepGamma(state, overflow)

            if self._out.shouldLog(t, self.cgp_iters):
                self._out.log(
                    f"    [cGP] iter {t:4d}  wl={wl_val:.4f}  "
                    f"den={den_energy:.4f}  ovf={overflow:.4f}  "
                    f"\u03bb={state['lambda_d']:.2e}  gamma={state['gamma']:.4f}")

            converged = self._checkConvergence(t, wl_val, overflow, max_den, state)
            self._out.writeIter(t, "cGP", wl_val, overflow, alpha, state["lambda_d"],
                                state["gamma"],
                                stop_reason=state["stop_reason"] if converged else "",
                                lambda_cong=state["lambda_cong_eff"])
            state["prev_wl"] = wl_val
            state["prev_overflow"] = overflow

            self._out.saveFrame(t + frame_offset, state["pos"], wl_val, den_energy,
                                overflow, state["lambda_d"], alpha, state["gamma"],
                                benchmark, state["num_macros"], phase="4: cGP")

            if converged:
                break

        self._out.log(f"  [cGP] done  iters={t + 1}  wl={state['prev_wl']:.4f}")
        self._out.banner(f"Phase 4: cGP  done (iters={t + 1})")
        return state["best_pos"]

    def _initState(self, benchmark, net_data):
        """Initialise all loop state variables."""
        num_macros = benchmark.num_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        canvas_diag = max(canvas_w, canvas_h)

        gamma = self.gamma or canvas_diag / 8.0
        max_step = canvas_diag * self.max_step_frac
        gamma_min = canvas_diag * self.gamma_min_frac

        fixed_mask = benchmark.macro_fixed[:num_macros].to(self.device)
        half_w = benchmark.macro_sizes[:num_macros, 0].to(self.device) / 2
        half_h = benchmark.macro_sizes[:num_macros, 1].to(self.device) / 2

        hm_margin_x = canvas_w * self.hard_macro_boundary_margin
        hm_margin_y = canvas_h * self.hard_macro_boundary_margin
        hm_mask_f = benchmark.get_hard_macro_mask()[:num_macros].float().to(self.device)
        extra_x = hm_mask_f * hm_margin_x
        extra_y = hm_mask_f * hm_margin_y
        clamp_lo_x = half_w + extra_x
        clamp_hi_x = canvas_w - half_w - extra_x
        clamp_lo_y = half_h + extra_y
        clamp_hi_y = canvas_h - half_h - extra_y

        precond = self._buildPreconditioner(benchmark, net_data, num_macros)

        # Set up frame directory before _initialPlacement so we can save an mIP frame.
        self._out.setupFrames(benchmark, net_data)
        self._out.openIterLog(benchmark)

        self._timer.start("mIP")
        pos, scatter_ids = self._initialPlacement(benchmark, net_data, num_macros,
                                                   canvas_w, canvas_h, half_w, half_h)
        self._timer.stop("mIP")
        if self.init_placement == "quadratic":
            self._out.saveMipFrame(pos, benchmark, num_macros)
        init_pos = pos.clone()

        scatter_lock_mask = torch.zeros(num_macros, dtype=torch.bool, device=self.device)
        if len(scatter_ids) > 0:
            scatter_lock_mask[scatter_ids] = True

        state = {
            "num_macros": num_macros,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "gamma": gamma,
            "gamma_base": gamma / 10.0,
            "max_step": max_step,
            "gamma_min": gamma_min,
            "fixed_mask": fixed_mask,
            "half_w": half_w,
            "half_h": half_h,
            "clamp_lo_x": clamp_lo_x,
            "clamp_hi_x": clamp_hi_x,
            "clamp_lo_y": clamp_lo_y,
            "clamp_hi_y": clamp_hi_y,
            "precond": precond,
            "pos": pos,
            "init_pos": init_pos,
            "lambda_d": 0.0,
            "lambda_cong_eff": 0.0,
            "lambda_cong_target": 0.0,
            "lambda_hm": self.lambda_hm_init,
            "hard_mask": benchmark.get_hard_macro_mask().to(self.device),
            "best_wl": float("inf"),
            "best_pos": pos.clone(),
            "prev_wl": float("inf"),
            "prev_overflow": float("inf"),
            "loss_history": [],
            "overflow_history": [],
            "ref_hpwl": 1.0,
            "overflow_ema": float("inf"),
            "density_mask": None,  # None = all macros contribute; set for phase 4
            "halo_size": self.halo_size,
            "hard_macro_weight": self.hard_macro_density_weight,
            "soft_macro_weight": self.soft_macro_density_weight,
            "stop_reason": "",
            "conv_life": None,
            "macro_locks": [],
            "active_lock_mask": None,
            "scatter_lock_mask": scatter_lock_mask,
        }

        self._initOptimizer(state, max_step, pos)
        self._logSetup(state)
        return state

    # -------------------------------------------------------------------
    # Dynamic macro locks
    # -------------------------------------------------------------------

    def _addMacroLock(self, state, mask, unlock_fn):
        """Register a dynamic macro lock active until unlock_fn(state) returns True.

        mask      : bool tensor [num_macros] — which macros to hold fixed
        unlock_fn : callable(state) -> bool  — return True to release the lock
        """
        state["macro_locks"].append({"mask": mask, "unlock_fn": unlock_fn})

    def _tickMacroLocks(self, state):
        """Evaluate all lock conditions once per iteration.

        Removes locks whose unlock_fn returns True (logging each release) and
        stores the OR of all remaining active masks in state["active_lock_mask"]
        (None when no locks are active).
        """
        still_locked = []
        for lock in state["macro_locks"]:
            if lock["unlock_fn"](state):
                n = int(lock["mask"].sum().item())
                self._out.log(
                    f"  [lock] released {n} macro(s)  lambda={state['lambda_d']:.2e}")
            else:
                still_locked.append(lock)
        state["macro_locks"] = still_locked
        if not still_locked:
            state["active_lock_mask"] = None
            return
        mask = still_locked[0]["mask"].clone()
        for lock in still_locked[1:]:
            mask = mask | lock["mask"]
        state["active_lock_mask"] = mask

    def _buildPreconditioner(self, benchmark, net_data, num_macros):
        """Build per-macro preconditioner: macro_area + net_degree."""
        if not self.use_precond:
            return None
        macro_area = (benchmark.macro_sizes[:num_macros, 0]
                      * benchmark.macro_sizes[:num_macros, 1]).float().to(self.device)
        net_degree = torch.zeros(num_macros, dtype=torch.float32, device=self.device)
        macro_ids_t = net_data["macro_ids"]
        is_macro_t = net_data["is_macro"].float()
        net_degree.scatter_add_(0, macro_ids_t.clamp(min=0), is_macro_t)
        return (macro_area + net_degree).clamp(min=1.0).unsqueeze(1)

    def _initialPlacement(self, benchmark, net_data, num_macros,
                          canvas_w, canvas_h, half_w, half_h):
        """Set initial macro positions (from benchmark, center scatter, or quadratic WL).

        Returns (pos, scatter_ids) where scatter_ids is a numpy int64 array of
        macro indices that were scatter-locked during mIP (empty if none).
        """
        import numpy as np
        pos = benchmark.macro_positions[:num_macros].clone().float().to(self.device)
        scatter_ids = np.empty(0, dtype=np.int64)
        if self.init_placement == "center":
            spread_r = self.init_spread ** 0.5
            cx, cy = canvas_w / 2.0, canvas_h / 2.0
            half_bx = canvas_w * spread_r / 2.0
            half_by = canvas_h * spread_r / 2.0
            pos[:, 0] = torch.empty(num_macros, device=self.device).uniform_(cx - half_bx, cx + half_bx)
            pos[:, 1] = torch.empty(num_macros, device=self.device).uniform_(cy - half_by, cy + half_by)
            pos[:, 0] = pos[:, 0].clamp(half_w, canvas_w - half_w)
            pos[:, 1] = pos[:, 1].clamp(half_h, canvas_h - half_h)
        elif self.init_placement == "quadratic":
            _qp = _importSibling("quadratic_placer")
            cfg = {"quad_b2b_iters": self.quad_b2b_iters,
                   "quad_net_size_threshold": self.quad_net_size_threshold,
                   "quad_anchor_fraction": self.quad_anchor_fraction,
                   "quad_scatter_n": self.quad_scatter_n,
                   "quad_scatter_fraction": self.quad_scatter_fraction,
                   "seed": self.seed,
                   "quad_scatter_section_count": self._section_count,
                   "quad_scatter_section_idx":   self._section_idx}
            self._out.log("  Quadratic WL init (mIP)...", flush=True)
            # quadratic_placer uses numpy/scipy and requires CPU tensors.
            cpu_net_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in net_data.items()}
            pos, scatter_ids = _qp.quadratic_init(cpu_net_data, benchmark, cfg)
            pos = pos.to(self.device)
            self._out.saveScatterIds(scatter_ids)
            self._out.log(f"  mIP done.")
        return pos, scatter_ids

    def _initOptimizer(self, state, max_step, pos):
        """Initialise optimizer-specific state variables."""
        if self.optimizer in ("bb_sgd", "nesterov"):
            state["alpha_k"] = self.alpha_init or max_step
            state["g_prev"] = None
            state["v_prev"] = None
        if self.optimizer == "nesterov":
            state["u_k"] = pos.clone()
            state["v_k"] = pos.clone()
            state["a_k"] = 1.0

    def _logSetup(self, state):
        """Log gradient descent parameters."""
        alpha0 = self.alpha_init or state["max_step"]
        if self.optimizer != "sgd":
            opt_info = (f"{self.optimizer}  alpha_init={alpha0:.4f}  "
                        f"max_step={state['max_step']:.4f}")
        else:
            opt_info = f"sgd  max_step={state['max_step']:.4f}"
        self._out.log(
            f"  Device: {self.device}  "
            f"Gradient descent: {self.max_iters} iters  "
            f"gamma0={state['gamma']:.3f}  gamma_min={state['gamma_min']:.4f}  "
            f"warmup={self.warmup_iters}  stop_overflow={self.stop_overflow}  "
            f"optimizer={opt_info}")

    # -------------------------------------------------------------------
    # Single iteration
    # -------------------------------------------------------------------

    def _stepGamma(self, state, overflow):
        if self.gamma_mode == "ovfw":
            coef = 10.0 ** ((overflow - 0.1) * 20.0 / 9.0 - 1.0)
            state["gamma"] = max(coef * state["gamma_base"], state["gamma_min"])
        else:
            state["gamma"] = max(state["gamma"] * self.gamma_decay, state["gamma_min"])

    def _runOneIteration(self, t, state, benchmark, net_data):
        """Execute one gradient descent iteration. Returns True if should stop."""
        self._tickMacroLocks(state)

        eval_pos = state["v_k"] if self.optimizer == "nesterov" else state["pos"]

        wl_grad, wl_val = self._computeWlGradient(eval_pos, net_data, state["gamma"])
        den_grad, den_energy, overflow, max_den = self._computeDenGradient(
            t, eval_pos, benchmark, state)
        cong_grad = self._computeCongGradient(t, eval_pos, net_data, benchmark, state, wl_grad, overflow)

        grad = self._combineGradients(wl_grad, den_grad, state, cong_grad)

        with torch.no_grad():
            grad[state["fixed_mask"]] = 0.0
            if state["active_lock_mask"] is not None:
                grad[state["active_lock_mask"]] = 0.0
            alpha = self._applyOptimizer(t, grad, state, eval_pos, wl_val)

            if torch.isnan(state["pos"]).any():
                self._out.log(f"  [NaN] iter {t}: restoring best snapshot")
                state["pos"] = state["best_pos"].clone()
                if self.optimizer == "nesterov":
                    state["u_k"] = state["pos"].clone()
                    state["v_k"] = state["pos"].clone()
                    state["a_k"] = 1.0
                return True

        self._trackBestWl(wl_val, overflow, state)
        if t % self.lambda_den_iters_per_update == 0:
            self._updateLambda(t, wl_val, wl_grad, den_grad, den_energy, state,
                               overflow=overflow)
        if t % self.gamma_iters_per_update == 0:
            self._stepGamma(state, overflow)
        # Decay hard-macro density boost toward 1.0
        if state["lambda_hm"] > 1.0:
            state["lambda_hm"] = max(1.0, state["lambda_hm"] * self.lambda_hm_decay)


        if self._out.shouldLog(t, self.max_iters):
            self._out.log(
                f"    iter {t:4d}  wl={wl_val:.4f}  den={den_energy:.4f}  "
                f"ovf={overflow:.4f}  \u03bb={state['lambda_d']:.2e}  "
                f"gamma={state['gamma']:.4f}")

        converged = self._checkConvergence(t, wl_val, overflow, max_den, state)
        self._out.writeIter(t, "mGP", wl_val, overflow, alpha, state["lambda_d"],
                            state["gamma"],
                            stop_reason=state["stop_reason"] if converged else "",
                            lambda_cong=state["lambda_cong_eff"])

        state["prev_wl"] = wl_val
        state["prev_overflow"] = overflow

        self._out.saveFrame(t, state["pos"], wl_val, den_energy, overflow,
                            state["lambda_d"], alpha, state["gamma"],
                            benchmark, state["num_macros"])
        return converged

    # -------------------------------------------------------------------
    # Gradient computation
    # -------------------------------------------------------------------

    def _computeWlGradient(self, eval_pos, net_data, gamma):
        """Compute WL gradient via autograd. Returns (grad, loss_value)."""
        pos_var = eval_pos.detach().requires_grad_(True)
        wl_loss = _waHpwl(pos_var, net_data, gamma)
        wl_loss.backward()
        return pos_var.grad.clone(), wl_loss.item()

    def _computeDenGradient(self, t, eval_pos, benchmark, state):
        """Compute density gradient (or zeros during warmup)."""
        need_density = (state["lambda_d"] > 0.0 or
                        (t == self.warmup_iters
                         and self.lambda_den_schedule == "hpwl"))
        if need_density:
            return _density.computeDensityGradient(
                self.density_method, eval_pos, benchmark, self.target_density,
                density_mask=state.get("density_mask"),
                halo_size=state.get("halo_size", 0.0),
                grid_rows=self.density_grid_rows,
                grid_cols=self.density_grid_cols,
                hard_macro_weight=state.get("hard_macro_weight", 1.0),
                soft_macro_weight=state.get("soft_macro_weight", 1.0))
        return (torch.zeros_like(state["pos"]), 0.0, float("inf"), float("inf"))

    def _computeCongGradient(self, t, eval_pos, net_data, benchmark, state, wl_grad, overflow=float("inf")):
        """Compute RUDY congestion gradient (or None when disabled/warmup/overflow too high).

        On first activation, the target weight is calibrated once against the WL gradient:
          lambda_cong_target = lambda_cong_init * wl_max / cong_max
        lambda_cong_eff then ramps from 1% of target toward target by lambda_cong_ramp
        each iteration, so congestion activates gradually rather than with a hard jolt.
        """
        if self.lambda_cong_init <= 0.0 or t < self.warmup_iters or overflow > self.cong_start_overflow:
            return None
        rudy = _congestion.compute_rudy_map(
            eval_pos, net_data,
            float(benchmark.canvas_width), float(benchmark.canvas_height),
            self.density_grid_rows, self.density_grid_cols,
        )
        rudy = (rudy - rudy.mean()).clamp(min=0.0)  # overflow-only: zero out under-capacity bins
        cong_grad = _density.computePoissonGradient(
            rudy, eval_pos, benchmark,
            self.density_grid_rows, self.density_grid_cols,
        )
        if state["lambda_cong_target"] == 0.0:
            wl_max   = wl_grad.abs().max().item()
            cong_max = cong_grad.abs().max().item()
            target   = self.lambda_cong_init * wl_max / (cong_max + 1e-8)
            state["lambda_cong_target"] = target
            state["lambda_cong_eff"]    = target * 0.01
            self._out.log(
                f"  cong activated: target={target:.3e}  "
                f"(wl_max={wl_max:.3e}  cong_max={cong_max:.3e})  ramp={self.lambda_cong_ramp}"
            )
        else:
            state["lambda_cong_eff"] = min(
                state["lambda_cong_eff"] * self.lambda_cong_ramp,
                state["lambda_cong_target"],
            )
        return cong_grad

    def _combineGradients(self, wl_grad, den_grad, state, cong_grad=None):
        """Combine WL and density gradients, apply preconditioner.

        Hard macros get an extra multiplier (lambda_hm) on their density
        gradient so they spread out before soft macros.
        """
        scaled_den = den_grad.clone()
        if state["lambda_hm"] > 1.0:
            scaled_den[state["hard_mask"]] *= state["lambda_hm"]
        grad = wl_grad + state["lambda_d"] * scaled_den
        if cong_grad is not None:
            grad = grad + state["lambda_cong_eff"] * cong_grad
        if state["precond"] is not None:
            grad = grad / state["precond"]
        return grad

    # -------------------------------------------------------------------
    # Optimizer step
    # -------------------------------------------------------------------

    def _applyOptimizer(self, t, grad, state, eval_pos, wl_val):
        """Dispatch to the selected optimizer. Returns alpha (step size)."""
        if self.optimizer in ("bb_sgd", "nesterov"):
            return self._stepBbBased(t, grad, state, eval_pos, wl_val)
        return self._stepSgd(grad, state)

    def _stepSgd(self, grad, state):
        """SGD with per-macro gradient clipping."""
        per_macro_norm = grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (state["max_step"] / per_macro_norm).clamp(max=1.0)
        clipped_grad = grad * scale
        alpha = (per_macro_norm * scale)[~state["fixed_mask"]].mean().item()

        state["pos"] = state["pos"].detach() - clipped_grad
        self._clampPositions(state)
        return alpha

    def _stepBbBased(self, t, grad, state, eval_pos, wl_val):
        """BB-SGD or Nesterov step with Barzilai-Borwein adaptive step size."""
        alpha_k = self._computeBbAlpha(grad, eval_pos, state)
        state["alpha_k"] = alpha_k

        step_vec = alpha_k * grad
        step_norm = step_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        step_vec = step_vec * (state["max_step"] / step_norm).clamp(max=1.0)

        if self.optimizer == "nesterov":
            self._stepNesterov(step_vec, state, wl_val)
        else:
            state["pos"] = state["pos"].detach() - step_vec
            self._clampPositions(state)

        return step_vec[~state["fixed_mask"]].norm(dim=1).mean().item()

    def _computeBbAlpha(self, grad, eval_pos, state):
        """Compute Barzilai-Borwein step size from gradient/position diffs."""
        g_flat = grad.reshape(-1)
        eval_flat = eval_pos.reshape(-1)

        if state["g_prev"] is not None:
            s_k = eval_flat - state["v_prev"]
            y_k = g_flat - state["g_prev"]
            # torch.dot delegates to cuBLAS sdot, which crashes with SIGFPE on some
            # CUDA driver versions. (a * b).sum() is identical but avoids cuBLAS.
            sy = (s_k * y_k).sum().item()
            yy = (y_k * y_k).sum().item()
            ss = (s_k * s_k).sum().item()
            if sy > 0.0 and yy > 1e-20:
                alpha = sy / yy                              # short BB
            elif ss > 1e-20 and yy > 1e-20:
                alpha = (ss ** 0.5) / (yy ** 0.5)           # Lipschitz fallback
            else:
                alpha = state["alpha_k"]
        else:
            alpha = state["alpha_k"]

        state["g_prev"] = g_flat.clone()
        state["v_prev"] = eval_flat.clone()
        return alpha

    def _stepNesterov(self, step_vec, state, wl_val):
        """Nesterov momentum update with restart on WL overshoot."""
        if (state["prev_wl"] < float("inf")
                and wl_val > state["prev_wl"] * 1.05):
            state["v_k"] = state["pos"].clone()
            state["a_k"] = 1.0

        a_k = state["a_k"]
        a_kp1 = (1.0 + (1.0 + 4.0 * a_k * a_k) ** 0.5) / 2.0
        coef = (a_k - 1.0) / a_kp1

        u_kp1 = state["v_k"].detach() - step_vec
        u_kp1[state["fixed_mask"]] = state["init_pos"][state["fixed_mask"]]
        if state["active_lock_mask"] is not None:
            u_kp1[state["active_lock_mask"]] = state["init_pos"][state["active_lock_mask"]]
        u_kp1[:, 0] = u_kp1[:, 0].clamp(min=state["clamp_lo_x"], max=state["clamp_hi_x"])
        u_kp1[:, 1] = u_kp1[:, 1].clamp(min=state["clamp_lo_y"], max=state["clamp_hi_y"])

        v_kp1 = u_kp1 + coef * (u_kp1 - state["u_k"])
        v_kp1[state["fixed_mask"]] = state["init_pos"][state["fixed_mask"]]
        if state["active_lock_mask"] is not None:
            v_kp1[state["active_lock_mask"]] = state["init_pos"][state["active_lock_mask"]]
        v_kp1[:, 0] = v_kp1[:, 0].clamp(min=state["clamp_lo_x"], max=state["clamp_hi_x"])
        v_kp1[:, 1] = v_kp1[:, 1].clamp(min=state["clamp_lo_y"], max=state["clamp_hi_y"])

        state["u_k"] = u_kp1
        state["v_k"] = v_kp1
        state["a_k"] = a_kp1
        state["pos"] = u_kp1

    def _clampPositions(self, state):
        """Clamp positions to canvas bounds and restore fixed/locked macros."""
        state["pos"][:, 0] = state["pos"][:, 0].clamp(
            min=state["clamp_lo_x"], max=state["clamp_hi_x"])
        state["pos"][:, 1] = state["pos"][:, 1].clamp(
            min=state["clamp_lo_y"], max=state["clamp_hi_y"])
        state["pos"][state["fixed_mask"]] = state["init_pos"][state["fixed_mask"]]
        if state["active_lock_mask"] is not None:
            state["pos"][state["active_lock_mask"]] = state["init_pos"][state["active_lock_mask"]]

    # -------------------------------------------------------------------
    # Lambda schedule
    # -------------------------------------------------------------------

    def _trackBestWl(self, wl_val, overflow, state):
        """Snapshot best position: only when overflow is below stop threshold and WL improves."""
        if overflow < self.stop_overflow and wl_val < state["best_wl"]:
            state["best_wl"] = wl_val
            state["best_pos"] = state["pos"].clone()

    def _updateLambda(self, t, wl_val, wl_grad, den_grad, den_energy, state,
                      overflow=None):
        """Update density weight lambda according to the schedule."""
        total_loss = wl_val + state["lambda_d"] * den_energy
        state["loss_history"].append(total_loss)

        if t < self.warmup_iters:
            return

        if state["lambda_d"] == 0.0:
            self._initLambda(wl_val, wl_grad, den_grad, state, overflow)
        else:
            self._rampLambda(t, wl_val, overflow, state)

    def _initLambda(self, wl_val, wl_grad, den_grad, state, overflow=None):
        """Auto-initialise lambda at warmup transition."""
        if self.lambda_den_schedule == "hpwl":
            wl_norm = wl_grad.norm(p=1).item()
            den_norm = den_grad.norm(p=1).item()
            state["lambda_d"] = self.lambda_den_init * wl_norm / (den_norm + 1e-8)
            state["ref_hpwl"] = wl_val
            state["overflow_ema"] = overflow if overflow is not None else float("inf")
        else:
            state["lambda_d"] = self.lambda_den_weight_init
            state["overflow_ema"] = float("inf")

        if (self.scatter_lock_mult > 0
                and state["scatter_lock_mask"].any()):
            thresh = self.scatter_lock_mult * state["lambda_d"]
            n = int(state["scatter_lock_mask"].sum().item())
            self._out.log(
                f"  [lock] locking {n} scatter macro(s) until "
                f"lambda >= {thresh:.2e}  (lambda_0={state['lambda_d']:.2e})")
            self._addMacroLock(
                state,
                state["scatter_lock_mask"],
                lambda s, t=thresh: s["lambda_d"] >= t,
            )

    def _rampLambda(self, t, wl_val, overflow, state):
        """Ramp lambda using overflow-feedback or geometric schedule.

        Overflow-feedback (hpwl mode):
          Uses an EMA of recent overflow (τ≈10 iters) as the reference.
          - Current overflow is >1% below EMA → spreading is actively working
            → hold lambda (mu=1.0). This prevents chaotic explosion once macros
            start separating.
          - Otherwise → ramp by lambda_pcof_upper.
            Macros are stuck; push the density force harder.

          Runs every iteration (no t%3 gate). When macros are stuck the ramp
          is 3× faster than the old every-3-iter approach, reaching the
          spreading threshold sooner.
        """
        if self.lambda_den_schedule == "hpwl":
            ema = state.get("overflow_ema", float("inf"))
            if overflow is not None and ema < float("inf"):
                state["overflow_ema"] = 0.9 * ema + 0.1 * overflow
                if overflow < state["overflow_ema"] * 0.99:
                    # Overflow is meaningfully below recent average — spreading is
                    # working; ramp gently (pcof_lower) so lambda still grows but
                    # doesn't overshoot.
                    mu = self.lambda_den_pcof_lower
                else:
                    mu = self.lambda_den_pcof_upper
            else:
                if overflow is not None:
                    state["overflow_ema"] = overflow
                mu = self.lambda_den_pcof_upper
        else:
            mu = self.lambda_den_max_step

        state["lambda_d"] = min(state["lambda_d"] * mu, self.lambda_den_max)


    # -------------------------------------------------------------------
    # Convergence checks
    # -------------------------------------------------------------------

    def _checkConvergence(self, t, wl_val, overflow, max_den, state):
        """Evaluate stopping criteria. Returns True if converged."""
        state["overflow_history"].append(overflow)

        if t < self.warmup_iters or t <= 100:
            return False

        # Divergence exits immediately (no countdown).
        if self.conv_divergence and self._checkDivergence(wl_val, state, t):
            return True

        # If countdown already started, tick it down.
        if state["conv_life"] is not None:
            state["conv_life"] -= 1
            if state["conv_life"] <= 0:
                self._out.log(f"  [converged] iter {t}: countdown elapsed")
                return True
            return False

        # Check criteria; on first trigger, start countdown instead of stopping.
        triggered = (
            (self.conv_lgamma     and self._checkLgamma(wl_val, overflow, state, t))
            or (self.conv_max_den  and self._checkMaxDensity(max_den, t, state))
            or (self.conv_plateau  and self._checkPlateau(overflow, state, t))
            or (self.conv_lambda_max and self._checkLambdaMax(overflow, state, t))
        )
        if triggered:
            state["conv_life"] = self.convergence_countdown
            self._out.log(f"  [converged] iter {t}: starting {self.convergence_countdown}-iter countdown")
        return False

    def _checkLgamma(self, wl_val, overflow, state, t):
        """Lgamma: overflow converged AND WL rising."""
        if overflow < self.stop_overflow and wl_val > state["prev_wl"]:
            self._out.log(f"  [converged] iter {t}: overflow {overflow:.4f} "
                          f"< {self.stop_overflow} and wl rising")
            state["stop_reason"] = "lgamma"
            return True
        return False

    def _checkMaxDensity(self, max_den, t, state):
        """Max-density: every bin already under target."""
        if max_den < self.target_density:
            self._out.log(f"  [converged] iter {t}: max_density {max_den:.4f} "
                          f"< target {self.target_density}")
            state["stop_reason"] = "max_density"
            return True
        return False

    def _checkPlateau(self, overflow, state, t):
        """Lsub plateau: moving-average loss flat."""
        w = self.plateau_window
        history = state["loss_history"]
        if (overflow < self.stop_overflow * 2 and len(history) >= 2 * w):
            cur_avg = sum(history[-w:]) / w
            prev_avg = sum(history[-2 * w:-w]) / w
            rel_change = abs(cur_avg - prev_avg) / (prev_avg + 1e-12)
            if rel_change < self.plateau_thresh:
                self._out.log(f"  [plateau] iter {t}: loss avg flat "
                              f"{cur_avg:.4f} \u2248 {prev_avg:.4f} "
                              f"(rel={rel_change:.4f})")
                state["stop_reason"] = "plateau"
                return True
        return False

    def _checkLambdaMax(self, overflow, state, t):
        """Lambda-max: lambda is pegged at ceiling AND overflow has stopped improving."""
        if state["lambda_d"] < self.lambda_den_max * 0.99:
            return False
        w = self.plateau_window
        oh = state["overflow_history"]
        if len(oh) >= 2 * w:
            cur_avg = sum(oh[-w:]) / w
            prev_avg = sum(oh[-2 * w:-w]) / w
            rel_change = abs(cur_avg - prev_avg) / (prev_avg + 1e-12)
            if rel_change < self.plateau_thresh:
                self._out.log(
                    f"  [lambda-max] iter {t}: lambda at max {state['lambda_d']:.2e}, "
                    f"overflow flat at {overflow:.4f}")
                state["stop_reason"] = "lambda_max"
                return True
        return False

    def _checkDivergence(self, wl_val, state, t):
        """Divergence: WL far exceeds best AND overflow rising."""
        dw = self.divergence_window
        oh = state["overflow_history"]
        if len(oh) >= dw and wl_val > state["best_wl"] * 5.0:
            window = oh[-dw:]
            trend = (window[-1] - window[0]) / (window[0] + 1e-12)
            if trend > 0.02:
                self._out.log(
                    f"  [diverged] iter {t}: wl {wl_val:.4f} > "
                    f"5\u00d7best {state['best_wl']:.4f}, "
                    f"ovf +{trend*100:.1f}% over {dw} iters")
                state["stop_reason"] = "divergence"
                return True
        return False
