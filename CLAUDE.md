# macro-place-challenge-2026 — Project Memory

## What this repo is
An EDA contest repository for VLSI macro placement. Submissions live in
`submissions/`. The contest harness in `macro_place/` evaluates each placer
on a suite of benchmarks and scores it on a proxy cost:

  proxy = 1.0×WL + 0.5×density + 0.5×congestion

## Config naming conventions

Config keys in `config.toml` that only apply under a specific parent setting
should be prefixed with that setting's value. This makes it immediately clear
at a glance which keys are active.

Examples:
- `center_init_spread` — only used when `initial_placement = "center"`
- `sa_T_init`, `sa_T_final`, `sa_steps_per_macro` — only used when
  `rotation_optimizer = "anneal"`

When adding a new config key that is conditional on another setting, follow
this pattern: `{parent_value}_{descriptive_name}`.

## How to run
```bash
# Always use uv run — the macro_place package is only in the project venv
uv run evaluate submissions/msears/placer.py        # single benchmark (ibm01)
uv run evaluate submissions/msears/placer.py --all  # all benchmarks
uv run python scripts/frames_to_gif.py              # render GIF from saved frames
```

## Shell environment note (for Claude)
This project lives in WSL (Ubuntu) but Claude runs in a Windows shell. All
commands must be prefixed with `wsl -e bash -c "..."` and use the full path to
uv. Example:
```bash
wsl -e bash -c "cd /home/msears/phd/macro-place-challenge-2026 && /home/msears/.local/bin/uv run evaluate submissions/msears/placer.py"
```

## Our submission: `submissions/msears/`

### Key files
| File | Purpose |
|------|---------|
| `submissions/msears/placer.py` | Main placer — CometPlacer class |
| `submissions/msears/density.py` | Density spreading forces (bell + electrostatic) |
| `submissions/msears/legalizer.py` | Macro legalization (bump + spiral); vectorized |
| `submissions/msears/output.py` | OutputManager: logging, frame recording, banners |
| `submissions/msears/quadratic_placer.py` | Quadratic WL solver (mIP); `select_scatter_ids` |
| `submissions/msears/config.toml` | Hyperparameter config (TOML, auto-discovered) |
| `scripts/frames_to_gif.py` | Offline GIF renderer from saved frame snapshots |

### Architecture
- **`_load_config()`** — reads `config.toml` via `tomllib` (stdlib, Python 3.11+)
- **`_load_plc(name)`** — loads PlacementCost object for IBM or NG45 benchmarks
- **`_build_net_data(benchmark, plc)`** — builds flat vectorized pin arrays (P pins total):
  - `macro_ids [P]` — benchmark macro index per pin (-1 = I/O port)
  - `offsets [P, 2]` — pin offsets from macro center (absolute pos for ports)
  - `net_ids [P]` — which net (0..K-1) each pin belongs to
  - `is_macro [P]` — bool mask distinguishing macro pins from port pins
- **`_wa_hpwl(pos, net_data, gamma)`** — vectorized smooth WA HPWL via scatter_add
- **`_density.compute_density_gradient(method, pos, benchmark, target_density)`** — density spreading force dispatcher (from `density.py`)
- **`PhaseTimer`** — stopwatch that wraps each placement phase; prints a timing table at end of each run
- **`CometPlacer.place(benchmark)`** — main entry point; calls `compute_proxy_cost` at end and saves `proxy_score.pt` via `OutputManager.saveProxyScore()`
- **`CometPlacer._runPlacementPipeline`** — orchestrates mGP → rotation → mLG → cGP; emits phase banners via `OutputManager.banner()`
- **`CometPlacer._gradient_place(benchmark, net_data)`** — gradient descent loop; calls `bumpLegalize`/`spiralLegalize` post-loop, saves `frame_legal.pt` via `OutputManager.saveLegalFrame()`

### Algorithm (current state)
WA HPWL + electrostatic density gradient descent, inspired by DREAMplace (Lin et al.
DAC 2019) and ePlace (Lu et al. DAC 2015).

Each iteration:
1. **WL gradient**: forward WA HPWL → `wl_loss.backward()` → `wl_grad [n, 2]`
2. **Density gradient**: `density.compute_density_gradient()` → `den_grad [n, 2]`  (skipped during warmup)
3. **Combined**: `grad = wl_grad + λ * den_grad`
4. Zero gradients for fixed macros
5. Per-macro gradient clipping (max_step)
6. Gradient step: `pos -= clipped_grad`
7. Clamp to canvas bounds, restore fixed macros
8. Ramp λ after warmup; decay gamma
9. **Convergence checks** (post-warmup, after iter 100):
   - **Lgamma** — `overflow < stop_overflow` AND `wl > prev_wl`: density converged, WL rising → stop
   - **Max-density** — `max_density < target_density`: every bin already under target → stop
   - **Lsub plateau** — moving avg of total loss barely improving over `plateau_window` iters → stop
   - **Divergence** — overflow trending up AND `wl > 2 × best_wl` → restore best position and stop

`density.compute_density_gradient()` now returns `(grad, energy, overflow_ratio, max_density)`.

### WA HPWL formula
```
WA_max_k = Σ[u_i · exp(u_i/γ)] / Σ[exp(u_i/γ)]   ≈ max(u_i)
WA_min_k = Σ[u_i · exp(-u_i/γ)] / Σ[exp(-u_i/γ)]  ≈ min(u_i)
loss = Σ_k [(WA_max_k - WA_min_k)_x + (WA_max_k - WA_min_k)_y]
```
Numerical stability: per-net shifts (not global) to prevent exp() underflow
when gamma is small. Uses `scatter_reduce_` with `reduce="amax"`.

### Density methods (`density.py`)

#### "bell" — Bell-shape kernel (older method, ref [14] in ePlace paper)
Smooth piecewise-quadratic C¹ approximation to macro-bin overlap.
Gradient flows through autograd (`torch.autograd.grad`). Local range only.
```
density_map[r,c] = Σ_m bell_y[m,r] * bell_x[m,c] / (bw*bh)
loss = Σ_{r,c} max(0, density_map[r,c] - target)²
```

#### "electrostatic" — ePlace Poisson FFT (current default)
Exact rectangular overlap density map → Poisson solve via 2D DCT → electric
field → bilinear interpolation at macro centers → analytic force. Long-range
global spreading. No autograd needed.

**Data flow:**
```
pos → _build_density_map_exact()  → density_map [rows, cols]
    → _poisson_fft_solve()        → Ex, Ey fields [rows, cols]
    → _bilinear_interp()          → fx, fy per macro [n]
    → den_grad [n, 2]
```

**Poisson solve (DREAMplace electric_potential.py):**
```
rho  = density_map / (bw * bh)
auv  = DCT2(rho)                           # 2D cosine transform
Ex   = IDXST_IDCT(auv * wu/(wu²+wv²)/2)   # sine in x, cosine in y
Ey   = IDCT_IDXST(auv * wv/(wu²+wv²)/2)   # cosine in x, sine in y
```
DCT2 basis encodes Neumann BCs (∂φ/∂n=0 at canvas edges).
2N-padding trick implements DCT2 via `torch.fft.rfft` — no extra libraries.
Spectral constants (expk, eigenvalues) cached per grid shape.

**Density map clamping (DREAMplace electric_overflow.py):**
Each macro clamped to `max(√2 · bin_size, node_size)` so every macro always
overlaps at least one bin. Density scaled by `original_area / clamped_area`.

### Frame export
When `record_frames = true` in config.toml, each iteration saves:
- `vis/frames/{benchmark}/frame_{t:05d}.pt` — positions, wl_loss, den_loss, lambda_d, gamma, sizes
- `vis/frames/{benchmark}/net_edges.pt` — net topology (saved once, not per-frame)
  - Contains: macro_ids, offsets, net_ids, is_macro, first_pin_idx
  - `first_pin_idx[k]` = flat index of driver pin of net k (for star rendering)

GIF renderer (`scripts/frames_to_gif.py`) draws:
- Hard macros: steelblue, soft macros: mediumseagreen, fixed: red
- Net lines: three layers — plain (black), IO-port nets (orange), highlighted macro nets (crimson)
- Proxy score overlay on final/legalized frame (bottom-left): proxy, WL, density, congestion, overlaps
- CLI flags: `--benchmark`, `--fps`, `--step`, `--dpi`, `--net-alpha`, `--no-nets`, `--highlight-io N`
- Config: `highlight_io_nets = N` highlights up to N nets connected to fixed I/O ports (0 = all)

## Benchmarks

### IBM (ICCAD04) — `external/MacroPlacement/Testcases/ICCAD04/{name}/`
Hard macros only. ibm01: 246 hard + 894 soft = 1140 total, 5993 nets.

### NG45 — `external/MacroPlacement/Flows/NanGate45/{design}/netlist/output_CT_Grouping/`
Mix of hard and soft. ariane133: 133 hard + 782 soft macros.

## Current results (ibm01)
```
# Pre-legalization greedy rotation, all 8 orientations, macro_sizes swap (current best):
proxy≈0.99  (0 overlaps)
```

## All-benchmark results (best config, 17 IBM benchmarks)
```
rank  avg_proxy   n_bench    params
1     1.4431      17         rotation_optimizer=greedy  ← last measured all-bench
```
Status: QUALIFIED (0 overlaps). ibm01 improved from ~1.04 to ~0.99 with rotation refactor.

## Rotation optimizer design

### Correct E/W orientation handling
- Orientations 0-3 (N/FN/S/FS) are mirror-only: footprint W×H unchanged.
- Orientations 4-7 (E/FE/W/FW) are 90° rotations: footprint physically becomes H×W.
- `plc.update_macro_orientation` only rotates **pin offsets** — it never swaps macro
  dimensions in the plc object.
- The harness overlap checker (`objective.py`) uses `benchmark.macro_sizes` directly,
  not plc dimensions. Same for the legalizer.
- Therefore: rotating to E/W **requires** swapping `benchmark.macro_sizes[m]` (w↔h)
  so the legalizer and harness both see the correct physical footprint.
- `_applyOrientationSizes(old_genes, new_genes, benchmark)` is the helper that does
  this correctly — only swaps when crossing the E/W boundary, safe to call repeatedly.

### Pipeline order
Rotation **must** run **before** legalization so the legalizer sees the correct
footprint. Running it after legalization produces physically inconsistent placements
(legalizer used W×H, harness checks against swapped H×W → detects overlaps).

Current order: mGP → rotation (pre-legalization) → mLG → cGP

### Periodic in-flight rotation (shelved)
Tried running greedy rotation every 10 iters during mGP with proper macro_sizes swaps.
Result: proxy hurt (~1.05 vs ~0.99 for pre-legalization greedy). The footprint changes
disrupt the density gradient too much mid-optimization. Shelved — `rotation_optimizer
= "greedy"` (pre-legalization single pass) remains the best approach found so far.

## Key technical concepts

### scatter_add pattern (core of _wa_hpwl)
- P pins total, K nets, M macros
- Gather: `pos[safe_ids]` — read macro positions in pin order
- Scatter: `zeros(K).scatter_add(0, net_ids, values)` — sum per net
- 20× faster than Python loop (2.88s → 0.146s per pass)

### Port handling trick
Ports have macro_id = -1 (no macro). Handled without branching:
  `safe_ids = macro_ids.clamp(min=0)`   (replaces -1 with 0)
  `pin_pos = pos[safe_ids] * is_macro.float() + offsets`
Port pins get `0 * pos[0] + absolute_pos = absolute_pos` ✓
Macro pins get `1 * pos[bidx] + offset` ✓
Gradient of pos[0] from port dummy entries = 0 (no effect) ✓

### Gradient split (WL vs density)
WL and density gradients are computed separately and summed:
- WL: `wl_loss.backward()` → autograd fills `pos_var.grad`
- Density: `density.compute_density_gradient()` returns analytic `den_grad`
- Combined: `grad = wl_grad + lambda_d * den_grad`
This is cleaner than a single combined loss because the electrostatic gradient
is computed analytically (no computation graph needed).

### DCT2 / DST-III via 2N-padding trick
Avoids external DCT libraries. Spectral constants cached per grid shape.

**Forward DCT2 (`_dct_2N`):** pad N→2N, rfft, multiply by `2*exp(-iπk/(2N))`, real part, divide by N.

**Inverse DCT2 (`_idct_2N`):** multiply by `2*exp(-iπk/(2N))`, pad, irfft, take samples `[1:N+1]`, multiply by N.

**DST-III (`_idxst_2N`):** computes `Σ x[k]*sin(πk*(2j+1)/(2N))` — the x-derivative of the DCT2 basis. Twiddle is `-2i*exp(-iπk/(2N))` = `complex(expk[:,1], -expk[:,0])`. Take irfft samples `[1:N+1]`.

**Key bug fixed:** original `_idxst_2N` used `expkp1 = 2*exp(+iπ(k+1)/(2N))` which produced a cosine output (always positive) instead of a sine (antisymmetric). This caused all macros to collapse to a corner. The fix: use `complex(expk[:,1], -expk[:,0])` (no separate expkp1 needed).

**Sign convention:** the potential φ has a maximum at density peaks (∇²φ=ρ with Neumann BCs). The repulsive spreading gradient is `den_grad = -E = +∇φ`, so `_density_gradient_electrostatic` returns `(-Ex, -Ey)` from the Poisson solver.

## Shelved / other branches

### `genetic_init` branch
GA for joint scatter-macro position + orientation search. Each chromosome encodes
orientations (scatter macros only) + initial (x, y) positions. Fitness = WL after
a full placement with scatter macros pinned at chromosome positions. Shelved because
the non-GA pipeline (greedy rotation + mLG) reaches ~0.99 proxy on ibm01 with 0
overlaps; GA adds significant runtime complexity for uncertain gain.

Key files on that branch:
- `submissions/msears/genetic.py` — `Chromosome`, `GeneticPlacer`, `PlacementEvalFn`
- `placer.py` — `_runGa`, `_buildFullEvalConfig`

### NOT yet implemented
- Nesterov momentum + BB step size
