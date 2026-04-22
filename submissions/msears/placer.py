"""
CometPlacer — Analytical WA HPWL + Density Gradient Descent

A PyTorch-based analytical placer inspired by DREAMplace (Lin et al. DAC 2019)
and ePlace (Lu et al. DAC 2015).

Placement phases:
  1. mIP  — macro initial placement (quadratic WL, center scatter, or benchmark default)
  2. mGP  — mixed global placement: WA HPWL + electrostatic density gradient descent
  2.5     — hard-macro spread: soft macros excluded from density, hard macros spread freely
  3. mLG  — macro legalization: bump or spiral push-out to remove hard-macro overlaps
  3.5     — post-legalization rotation optimizer (greedy or SA)
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
_genetic = _importSibling("genetic")
_rotation = _importSibling("rotation")


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

    return _flattenNets(plc, pin_to_macro, port_pos)


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


def _flattenNets(plc, pin_to_macro, port_pos):
    """Flatten all nets into parallel pin arrays."""
    macro_ids_list = []
    ox_list = []
    oy_list = []
    net_ids_list = []
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
    offsets = torch.tensor(list(zip(ox_list, oy_list)), dtype=torch.float32)
    net_ids = torch.tensor(net_ids_list, dtype=torch.long)
    is_macro = (macro_ids >= 0)

    return {
        "macro_ids": macro_ids,
        "offsets": offsets,
        "net_ids": net_ids,
        "is_macro": is_macro,
        "num_pins": len(macro_ids_list),
        "num_nets": net_idx,
        "num_skipped": num_skipped,
    }


# ---------------------------------------------------------------------------
# Vectorized WA HPWL
# ---------------------------------------------------------------------------

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
    """

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

        self._readAlgorithmParams(params)
        self._readMethodParams(params)
        self._readLambdaParams(params)
        self._readConvergenceParams(params)
        self._readPlacementParams(params)
        self._out = _output.OutputManager(output_cfg)
        self._ga_config: dict = cfg.get("ga", {})

        # Populated after each placement; consumed by the GA fitness oracle.
        self.last_metrics: dict = {}

        use_gpu = cfg.get("params", {}).get("use_gpu", True)
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    def _readAlgorithmParams(self, p):
        self.max_iters = p.get("max_iters", 2000)
        self.soft_place_iters = p.get("soft_place_iters", 1000)
        self.seed = p.get("seed", 42)
        self.deterministic = _asBool(p.get("deterministic", False))
        self.gamma = None if p.get("gamma", "auto") == "auto" else float(p["gamma"])
        self.max_step = None if p.get("max_step", "auto") == "auto" else float(p["max_step"])
        self.gamma_decay = p.get("gamma_decay", 0.98)
        self.gamma_min_frac = p.get("gamma_min_frac", 1 / 150)

    def _readMethodParams(self, p):
        self.density_method = p.get("density_method", "electrostatic")
        self.legalization = p.get("legalization", "none")
        self.soft_place = _asBool(p.get("soft_place", False))
        self.hard_spread = _asBool(p.get("hard_spread", False))
        self.hard_spread_iters = p.get("hard_spread_iters", 50)
        self.halo_size = float(p.get("halo_size", 0.0))
        self.halo_legalize = float(p.get("halo_legalize", self.halo_size))
        self.use_precond = _asBool(p.get("use_preconditioner", True))
        self.optimizer = p.get("optimizer", "sgd")
        self.alpha_init = None if p.get("alpha_init", "auto") == "auto" else float(p["alpha_init"])
        self.soft_place_iters = p.get("soft_place_iters", 1000)
        self.rotation_optimizer  = p.get("rotation_optimizer", "none")   # none/greedy/anneal/periodic/genetic
        self.rotation_passes     = p.get("rotation_passes", 1)            # greedy: full passes over all macros
        self.rotation_period     = p.get("rotation_period", 20)           # periodic: iters between rotation calls
        self.sa_T_init           = p.get("sa_T_init", 1e-4)
        self.sa_T_final          = p.get("sa_T_final", 1e-7)
        self.sa_steps_per_macro  = p.get("sa_steps_per_macro", 200)
        _cand_map = {
            "NS":       (0, 2),
            "no-swap":  (0, 1, 2, 3),
            "all":      (0, 1, 2, 3, 4, 5, 6, 7),
        }
        self.rotation_candidates = _cand_map[p.get("rotation_candidates", "all")]

    def _readLambdaParams(self, p):
        self.lambda_schedule = p.get("lambda_schedule", "hpwl")
        self.density_weight = p.get("density_weight", 8e-4)
        self.lambda_pcof_upper = p.get("lambda_pcof_upper", 1.05)
        self.lambda_pcof_lower = p.get("lambda_pcof_lower", 0.95)
        self.density_weight_init = p.get("density_weight_init", 1e-5)
        self.density_weight_max_step = p.get("density_weight_max_step", 1.04)
        self.warmup_iters = p.get("warmup_iters", 20)
        self.density_weight_max = p.get("density_weight_max", 5000.0)
        self.target_density = p.get("target_density", 0.5)
        # Hard-macro density boost: multiplies density gradient for hard macros
        # only, starts at lambda_hm_init and decays toward 1.0 each iteration.
        self.lambda_hm_init = p.get("lambda_hm_init", 3.0)
        self.lambda_hm_decay = p.get("lambda_hm_decay", 0.995)

    def _readConvergenceParams(self, p):
        self.stop_overflow = p.get("stop_overflow", 0.1)
        self.plateau_window = p.get("plateau_window", 10)
        self.plateau_thresh = p.get("plateau_threshold", 0.001)
        self.divergence_window = p.get("divergence_window", 50)

    def _readPlacementParams(self, p):
        self.init_placement = p.get("initial_placement", "none")
        self.init_spread = p.get("initial_spread", 0.01)
        self.quad_b2b_iters = p.get("quad_b2b_iters", 3)
        self.quad_net_size_threshold = p.get("quad_net_size_threshold", 50)

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

        pos = self._runPlacementPipeline(benchmark, net_data)

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
        pos_cpu      = pos.detach().cpu()
        work_offs_dev = net_data["offsets"]                  # may be on CUDA
        work_offs_cpu = work_offs_dev.cpu().clone()

        mode = self.rotation_optimizer
        if mode == "greedy":
            n = _rotation.greedy_rotate(
                pos_cpu, work_offs_cpu, net_data_base,
                genes, benchmark, gamma=gamma,
                n_passes=self.rotation_passes,
                candidates=candidates,
            )
            self._out.log(f"  Rotation (greedy, {self.rotation_passes} pass, "
                          f"cands={candidates}): {n} macro(s) flipped")
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
            self._out.log(
                f"  Rotation (SA, {n_steps} steps, T {self.sa_T_init:.1e}→{self.sa_T_final:.1e}, "
                f"cands={candidates}): {n} accepted"
            )

        # Sync rotated offsets back to the original device tensor.
        work_offs_dev.copy_(work_offs_cpu.to(work_offs_dev.device))

        # Swap macro_sizes w↔h for E/W-oriented macros (orientations 4-7 transpose).
        for m in range(benchmark.num_hard_macros):
            if int(genes[m]) >= 4:
                w = benchmark.macro_sizes[m, 0].clone()
                benchmark.macro_sizes[m, 0] = benchmark.macro_sizes[m, 1]
                benchmark.macro_sizes[m, 1] = w

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
    # GA helpers
    # -------------------------------------------------------------------

    def _runGa(self, benchmark, net_data_base: dict) -> tuple[dict, object, object]:
        """
        Run the GA outer loop and return (net_data_mod, bench_mod, best_genes).

        The GA evaluates curtailed placements (no legalization, fewer iters,
        quiet output) via a lambda that instantiates a fresh CometPlacer with
        a stripped-down config.
        """
        curtailed_cfg = self._buildCurtailedConfig()

        def _eval(bench_mod, nd_mod):
            p = CometPlacer(curtailed_cfg)
            p.placeWithData(bench_mod, nd_mod)
            return p.last_metrics

        ga = _genetic.GeneticPlacer(self._ga_config)
        self._out.log(
            f"  GA: {ga.cfg}  "
            f"(curtailed_iters={ga.cfg.curtailed_iters})"
        )
        best_chrom = ga.run(benchmark, net_data_base, _eval)
        net_data_mod, bench_mod = best_chrom.applyToPlacement(net_data_base, benchmark)
        return net_data_mod, bench_mod, best_chrom.genes.clone()

    def _buildCurtailedConfig(self) -> dict:
        """
        Build a config dict for GA evaluation runs:
        - Inherits all [params] from the current placer (same density method,
          optimizer, lambda schedule, etc.)
        - Overrides max_iters with ga.curtailed_iters
        - Disables phase 2.5 (hard spread) and phase 4 (cGP) and legalization
        - Silences all output and frame recording
        """
        base_cfg = _loadConfig()
        params   = dict(base_cfg.get("params", {}))

        params["max_iters"]    = self._ga_config.get("ga_curtailed_iters", 300)
        params["hard_spread"]  = False
        params["soft_place"]   = False
        params["legalization"] = "none"

        return {
            "params": params,
            "output": {
                "quiet":             True,
                "record_frames":     False,
                "record_iterations": False,
                "log_every":         999999,
            },
            "ga": {},  # no nested GA in curtailed runs
        }

    # -------------------------------------------------------------------
    # Top-level entry point
    # -------------------------------------------------------------------

    def place(self, benchmark):
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

        _output.printBenchmarkInfo(benchmark, self.density_method)

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

        # Base net_data (all-N offsets) and gene state — used by rotation optimizer.
        net_data_base   = net_data
        current_genes   = torch.zeros(benchmark.num_hard_macros, dtype=torch.int8)

        # ── Orientation pre-search (GA or genetic mode) ───────────────────
        ga_active = (_asBool(self._ga_config.get("ga_enable", False))
                     or self.rotation_optimizer == "genetic")
        if ga_active:
            net_data, benchmark, current_genes = self._runGa(benchmark, net_data_base)

        pos = self._runPlacementPipeline(benchmark, net_data, net_data_base, current_genes, plc)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:benchmark.num_macros] = pos

        macro_dat = Path(self._out.frames_dir).parent / "data" / benchmark.name / "macros.dat"
        _output.writeMacroDat(pos, benchmark, net_data, plc, macro_dat)
        self._out.log(f"  Macro HPWL      -> {macro_dat}")
        self._out.log(f"  Total time: {time.time()-start_time:.1f}s")
        return full_pos

    # -------------------------------------------------------------------
    # Gradient descent loop
    # -------------------------------------------------------------------

    def _runPlacementPipeline(self, benchmark, net_data, net_data_base=None, current_genes=None, plc=None):
        """Orchestrates all placement phases: mGP → hard spread → mLG → rotation → cGP."""
        net_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in net_data.items()}

        state = self._initState(benchmark, net_data)
        rot_genes = (current_genes.clone()
                     if current_genes is not None
                     else torch.zeros(benchmark.num_hard_macros, dtype=torch.int8))

        # Phase 2: mGP
        last_t = self._mixedGlobalPlacement(state, benchmark, net_data, rot_genes, net_data_base)

        # Phase 2.5: hard-macro spread
        self._out.log(f"  hard_spread={self.hard_spread}  "
                      f"hard_spread_iters={self.hard_spread_iters}")
        if self.hard_spread and benchmark.num_soft_macros > 0:
            last_t = self._hardMacroSpread(state, benchmark, net_data, frame_offset=last_t + 1)

        self._out.close()


        # Phase 3.5: pre-legalization rotation optimizer
        if self.rotation_optimizer in ("greedy", "anneal") and net_data_base is not None:
            self._runRotationOptimizer(
                state["pos"], net_data, net_data_base, rot_genes, benchmark,
                state["gamma"], plc
            )

        # Phase 3: mLG
        self._macroLegalization(state, last_t, benchmark)

        # Phase 4: cGP
        if self.soft_place and benchmark.num_soft_macros > 0:
            state["pos"] = self._cellGlobalPlacement(state["pos"], benchmark, net_data,
                                                     frame_offset=last_t + 1)
            self._out.saveLegalFrame(last_t, state["pos"], state["prev_wl"],
                                     state["gamma"], benchmark, state["num_macros"],
                                     phase="3: mLG")

        self.last_metrics = {
            "wl": float(state["prev_wl"]),
            "overflow": float(
                state["overflow_history"][-1] if state["overflow_history"] else 1.0
            ),
        }
        return state["pos"].cpu()

    def _mixedGlobalPlacement(self, state, benchmark, net_data, rot_genes, net_data_base=None):
        """Phase 2 (mGP): WA HPWL + electrostatic density gradient descent main loop."""
        for t in range(self.max_iters):
            should_stop = self._runOneIteration(t, state, benchmark, net_data)
            if should_stop:
                break

            if (self.rotation_optimizer == "periodic"
                    and net_data_base is not None
                    and t > self.warmup_iters
                    and t % self.rotation_period == 0):
                n = _rotation.greedy_rotate(
                    state["pos"], net_data["offsets"], net_data_base,
                    rot_genes, benchmark, gamma=state["gamma"],
                )
                if n:
                    self._out.log(f"    [rotation t={t}] {n} macro(s) flipped")
        return t

    def _macroLegalization(self, state, last_t, benchmark):
        """Phase 3 (mLG): remove hard-macro overlaps via bump or spiral push-out."""
        if self.legalization == "bump":
            self._out.log("  Legalizing (pairwise bump)...")
            state["pos"] = _legalizer.bumpLegalize(
                state["pos"].cpu(), benchmark, halo_size=self.halo_legalize
            ).to(self.device)
            self._out.saveLegalFrame(last_t, state["pos"], state["prev_wl"],
                                     state["gamma"], benchmark, state["num_macros"])
        elif self.legalization == "spiral":
            self._out.log("  Legalizing (spiral push-out)...")
            state["pos"] = _legalizer.spiralLegalize(
                state["pos"].cpu(), benchmark
            ).to(self.device)
            self._out.saveLegalFrame(last_t, state["pos"], state["prev_wl"],
                                     state["gamma"], benchmark, state["num_macros"])

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
        state["lambda_d"] = self.density_weight

        self._out.log(
            f"  Phase 2.5 (hard spread): {self.hard_spread_iters} iters — "
            f"soft macros excluded from density map  "
            f"lambda reset {prev_lambda:.2e} -> {self.density_weight:.2e}")

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

            self._trackBestWl(wl_val, state)
            state["gamma"] = max(state["gamma"] * self.gamma_decay,
                                 state["gamma_min"])

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

        # Restart gamma and step size fresh for this phase
        gamma = canvas_diag / 8.0
        gamma_min = canvas_diag * self.gamma_min_frac
        max_step = self.max_step or canvas_diag * 0.005

        state = {
            "num_macros": num_macros,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "gamma": gamma,
            "max_step": max_step,
            "gamma_min": gamma_min,
            "fixed_mask": fixed_mask,
            "half_w": half_w,
            "half_h": half_h,
            "precond": self._buildPreconditioner(benchmark, net_data, num_macros),
            "pos": pos.clone(),
            "init_pos": pos.clone(),  # _clampPositions restores fixed macros to this
            "lambda_d": 0.0,
            "lambda_hm": 1.0,         # no hard-macro boost; they're fixed anyway
            "hard_mask": benchmark.get_hard_macro_mask().to(self.device),
            "best_wl": float("inf"),
            "best_pos": pos.clone(),
            "prev_wl": float("inf"),
            "prev_overflow": float("inf"),
            "loss_history": [],
            "overflow_history": [],
            "ref_hpwl": 1.0,
            "overflow_ema": float("inf"),
        }

        # Scatter soft macros uniformly across the canvas as a fresh start.
        # Hard macro positions in state["pos"] are already locked via fixed_mask
        # and will be restored each iteration by _clampPositions, so overwriting
        # the soft rows here is safe.
        #num_hard = benchmark.num_hard_macros
        #soft_hw = half_w[num_hard:]
        #soft_hh = half_h[num_hard:]
        #state["pos"][num_hard:, 0] = torch.empty(benchmark.num_soft_macros).uniform_(0, 1) \
        #    * (canvas_w - 2 * soft_hw.mean()) + soft_hw.mean()
        #state["pos"][num_hard:, 1] = torch.empty(benchmark.num_soft_macros).uniform_(0, 1) \
        #    * (canvas_h - 2 * soft_hh.mean()) + soft_hh.mean()
        #state["pos"][num_hard:, 0] = state["pos"][num_hard:, 0].clamp(
        #    soft_hw, canvas_w - soft_hw)
        #state["pos"][num_hard:, 1] = state["pos"][num_hard:, 1].clamp(
        #    soft_hh, canvas_h - soft_hh)
        #state["best_pos"] = state["pos"].clone()

        self._out.log(
            f"  Phase 4 (cGP): {benchmark.num_soft_macros} soft macros free  "
            f"max_iters={self.soft_place_iters}  warmup={self.warmup_iters}  "
            f"gamma0={gamma:.3f}")

        for t in range(self.soft_place_iters):
            eval_pos = state["pos"]

            wl_grad, wl_val = self._computeWlGradient(eval_pos, net_data, state["gamma"])
            den_grad, den_energy, overflow, max_den = self._computeDenGradient(
                t, eval_pos, benchmark, state)

            grad = self._combineGradients(wl_grad, den_grad, state)

            with torch.no_grad():
                grad[fixed_mask] = 0.0
                alpha = self._stepSgd(grad, state)

            self._trackBestWl(wl_val, state)
            self._updateLambda(t, wl_val, wl_grad, den_grad, den_energy, state,
                               overflow=overflow)
            state["gamma"] = max(state["gamma"] * self.gamma_decay, gamma_min)

            if self._out.shouldLog(t, self.soft_place_iters):
                self._out.log(
                    f"    [cGP] iter {t:4d}  wl={wl_val:.4f}  "
                    f"den={den_energy:.4f}  ovf={overflow:.4f}  "
                    f"\u03bb={state['lambda_d']:.2e}  gamma={state['gamma']:.4f}")

            converged = self._checkConvergence(t, wl_val, overflow, max_den, state)
            state["prev_wl"] = wl_val
            state["prev_overflow"] = overflow

            self._out.saveFrame(t + frame_offset, state["pos"], wl_val, den_energy,
                                overflow, state["lambda_d"], alpha, state["gamma"],
                                benchmark, state["num_macros"], phase="4: cGP")

            if converged:
                break

        self._out.log(f"  [cGP] done  iters={t + 1}  wl={state['prev_wl']:.4f}")
        return state["pos"]

    def _initState(self, benchmark, net_data):
        """Initialise all loop state variables."""
        num_macros = benchmark.num_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        canvas_diag = max(canvas_w, canvas_h)

        gamma = self.gamma or canvas_diag / 8.0
        max_step = self.max_step or canvas_diag * 0.005
        gamma_min = canvas_diag * self.gamma_min_frac

        fixed_mask = benchmark.macro_fixed[:num_macros].to(self.device)
        half_w = benchmark.macro_sizes[:num_macros, 0].to(self.device) / 2
        half_h = benchmark.macro_sizes[:num_macros, 1].to(self.device) / 2

        precond = self._buildPreconditioner(benchmark, net_data, num_macros)

        # Set up frame directory before _initialPlacement so we can save an mIP frame.
        self._out.setupFrames(benchmark, net_data)
        self._out.openIterLog(benchmark)

        pos = self._initialPlacement(benchmark, net_data, num_macros,
                                    canvas_w, canvas_h, half_w, half_h)
        if self.init_placement == "quadratic":
            self._out.saveMipFrame(pos, benchmark, num_macros)
        init_pos = pos.clone()

        state = {
            "num_macros": num_macros,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "gamma": gamma,
            "max_step": max_step,
            "gamma_min": gamma_min,
            "fixed_mask": fixed_mask,
            "half_w": half_w,
            "half_h": half_h,
            "precond": precond,
            "pos": pos,
            "init_pos": init_pos,
            "lambda_d": 0.0,
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
        }

        self._initOptimizer(state, max_step, pos)
        self._logSetup(state)
        return state

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
        """Set initial macro positions (from benchmark, center scatter, or quadratic WL)."""
        pos = benchmark.macro_positions[:num_macros].clone().float().to(self.device)
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
                   "quad_net_size_threshold": self.quad_net_size_threshold}
            self._out.log("  Quadratic WL init (mIP)...", flush=True)
            # quadratic_placer uses numpy/scipy and requires CPU tensors.
            cpu_net_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in net_data.items()}
            pos = _qp.quadratic_init(cpu_net_data, benchmark, cfg).to(self.device)
            self._out.log(f"  mIP done.")
        return pos

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

    def _runOneIteration(self, t, state, benchmark, net_data):
        """Execute one gradient descent iteration. Returns True if should stop."""
        eval_pos = state["v_k"] if self.optimizer == "nesterov" else state["pos"]

        wl_grad, wl_val = self._computeWlGradient(eval_pos, net_data, state["gamma"])
        den_grad, den_energy, overflow, max_den = self._computeDenGradient(
            t, eval_pos, benchmark, state)

        grad = self._combineGradients(wl_grad, den_grad, state)

        with torch.no_grad():
            grad[state["fixed_mask"]] = 0.0
            alpha = self._applyOptimizer(t, grad, state, eval_pos, wl_val)

            if torch.isnan(state["pos"]).any():
                self._out.log(f"  [NaN] iter {t}: restoring best snapshot")
                state["pos"] = state["best_pos"].clone()
                if self.optimizer == "nesterov":
                    state["u_k"] = state["pos"].clone()
                    state["v_k"] = state["pos"].clone()
                    state["a_k"] = 1.0
                return True

        self._trackBestWl(wl_val, state)
        self._updateLambda(t, wl_val, wl_grad, den_grad, den_energy, state,
                           overflow=overflow)
        state["gamma"] = max(state["gamma"] * self.gamma_decay, state["gamma_min"])
        # Decay hard-macro density boost toward 1.0
        if state["lambda_hm"] > 1.0:
            state["lambda_hm"] = max(1.0, state["lambda_hm"] * self.lambda_hm_decay)

        self._out.writeIter(t, wl_val, overflow, alpha, state["lambda_d"],
                            state["gamma"])
        if self._out.shouldLog(t, self.max_iters):
            self._out.log(
                f"    iter {t:4d}  wl={wl_val:.4f}  den={den_energy:.4f}  "
                f"ovf={overflow:.4f}  \u03bb={state['lambda_d']:.2e}  "
                f"gamma={state['gamma']:.4f}")

        converged = self._checkConvergence(t, wl_val, overflow, max_den, state)

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
                         and self.lambda_schedule == "hpwl"))
        if need_density:
            return _density.computeDensityGradient(
                self.density_method, eval_pos, benchmark, self.target_density,
                density_mask=state.get("density_mask"),
                halo_size=state.get("halo_size", 0.0))
        return (torch.zeros_like(state["pos"]), 0.0, float("inf"), float("inf"))

    def _combineGradients(self, wl_grad, den_grad, state):
        """Combine WL and density gradients, apply preconditioner.

        Hard macros get an extra multiplier (lambda_hm) on their density
        gradient so they spread out before soft macros.
        """
        scaled_den = den_grad.clone()
        if state["lambda_hm"] > 1.0:
            scaled_den[state["hard_mask"]] *= state["lambda_hm"]
        grad = wl_grad + state["lambda_d"] * scaled_den
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
        u_kp1[:, 0] = u_kp1[:, 0].clamp(
            min=state["half_w"], max=state["canvas_w"] - state["half_w"])
        u_kp1[:, 1] = u_kp1[:, 1].clamp(
            min=state["half_h"], max=state["canvas_h"] - state["half_h"])

        v_kp1 = u_kp1 + coef * (u_kp1 - state["u_k"])
        v_kp1[state["fixed_mask"]] = state["init_pos"][state["fixed_mask"]]
        v_kp1[:, 0] = v_kp1[:, 0].clamp(
            min=state["half_w"], max=state["canvas_w"] - state["half_w"])
        v_kp1[:, 1] = v_kp1[:, 1].clamp(
            min=state["half_h"], max=state["canvas_h"] - state["half_h"])

        state["u_k"] = u_kp1
        state["v_k"] = v_kp1
        state["a_k"] = a_kp1
        state["pos"] = u_kp1

    def _clampPositions(self, state):
        """Clamp positions to canvas bounds and restore fixed macros."""
        state["pos"][:, 0] = state["pos"][:, 0].clamp(
            min=state["half_w"], max=state["canvas_w"] - state["half_w"])
        state["pos"][:, 1] = state["pos"][:, 1].clamp(
            min=state["half_h"], max=state["canvas_h"] - state["half_h"])
        state["pos"][state["fixed_mask"]] = state["init_pos"][state["fixed_mask"]]

    # -------------------------------------------------------------------
    # Lambda schedule
    # -------------------------------------------------------------------

    def _trackBestWl(self, wl_val, state):
        """Track best WL and snapshot position."""
        if wl_val < state["best_wl"]:
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
        if self.lambda_schedule == "hpwl":
            wl_norm = wl_grad.norm(p=1).item()
            den_norm = den_grad.norm(p=1).item()
            state["lambda_d"] = self.density_weight * wl_norm / (den_norm + 1e-8)
            state["ref_hpwl"] = wl_val
            state["overflow_ema"] = overflow if overflow is not None else float("inf")
        else:
            state["lambda_d"] = self.density_weight_init
            state["overflow_ema"] = float("inf")

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
        if self.lambda_schedule == "hpwl":
            ema = state.get("overflow_ema", float("inf"))
            if overflow is not None and ema < float("inf"):
                state["overflow_ema"] = 0.9 * ema + 0.1 * overflow
                if overflow < state["overflow_ema"] * 0.99:
                    # Overflow is meaningfully below recent average — spreading is
                    # working; ramp gently (pcof_lower) so lambda still grows but
                    # doesn't overshoot.
                    mu = self.lambda_pcof_lower
                else:
                    mu = self.lambda_pcof_upper
            else:
                if overflow is not None:
                    state["overflow_ema"] = overflow
                mu = self.lambda_pcof_upper
        else:
            mu = self.density_weight_max_step

        state["lambda_d"] = min(state["lambda_d"] * mu, self.density_weight_max)

    # -------------------------------------------------------------------
    # Convergence checks
    # -------------------------------------------------------------------

    def _checkConvergence(self, t, wl_val, overflow, max_den, state):
        """Evaluate four stopping criteria. Returns True if converged."""
        state["overflow_history"].append(overflow)

        if t < self.warmup_iters or t <= 100:
            return False

        if self._checkLgamma(wl_val, overflow, state, t):
            return True
        if self._checkMaxDensity(max_den, t):
            return True
        if self._checkPlateau(overflow, state, t):
            return True
        if self._checkLambdaMax(overflow, state, t):
            return True
        if self._checkDivergence(wl_val, state, t):
            return True
        return False

    def _checkLgamma(self, wl_val, overflow, state, t):
        """Lgamma: overflow converged AND WL rising."""
        if overflow < self.stop_overflow and wl_val > state["prev_wl"]:
            self._out.log(f"  [converged] iter {t}: overflow {overflow:.4f} "
                          f"< {self.stop_overflow} and wl rising")
            return True
        return False

    def _checkMaxDensity(self, max_den, t):
        """Max-density: every bin already under target."""
        if max_den < self.target_density:
            self._out.log(f"  [converged] iter {t}: max_density {max_den:.4f} "
                          f"< target {self.target_density}")
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
                return True
        return False

    def _checkLambdaMax(self, overflow, state, t):
        """Lambda-max: lambda is pegged at ceiling AND overflow has stopped improving."""
        if state["lambda_d"] < self.density_weight_max * 0.99:
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
                return True
        return False
