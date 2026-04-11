# macro-place-challenge-2026 — Project Memory

## What this repo is
An EDA contest repository for VLSI macro placement. Submissions live in
`submissions/`. The contest harness in `macro_place/` evaluates each placer
on a suite of benchmarks and scores it on a proxy cost:

  proxy = 1.0×WL + 0.5×density + 0.5×congestion

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
| `submissions/msears/placer.py` | Main placer — MSPlacer class |
| `submissions/msears/density.py` | Density spreading forces (bell + electrostatic) |
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
- **`MSPlacer.place(benchmark)`** — main entry point called by contest harness
- **`MSPlacer._gradient_place(benchmark, net_data)`** — gradient descent loop

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
- Net lines: black, alpha=0.05, star topology (driver → each sink)
- CLI flags: `--benchmark`, `--fps`, `--step`, `--dpi`, `--net-alpha`, `--no-nets`

## Benchmarks

### IBM (ICCAD04) — `external/MacroPlacement/Testcases/ICCAD04/{name}/`
Hard macros only. ibm01: 246 hard + 894 soft = 1140 total, 5993 nets.

### NG45 — `external/MacroPlacement/Flows/NanGate45/{design}/netlist/output_CT_Grouping/`
Mix of hard and soft. ariane133: 133 hard + 782 soft macros.

## Current results (ibm01)
```
# WL-only (no density force):
proxy=4.65  (wl=0.021 den=3.996 cong=5.251)  INVALID (12892 overlaps)

# Bell-shape density (warmup=50, density_weight_init=1e-5, ramp=1.05×):
proxy=2.70  (wl=0.062 den=2.297 cong=2.973)  INVALID (816 overlaps)  [2.5s]

# Electrostatic density (ramp=1.04×, max_lambda=100):
proxy=1.04  (wl=0.073 den=0.743 cong=1.192)  INVALID (142 overlaps)  [1.8s]
```

## All-benchmark results (electrostatic, ramp=1.04×)
```
Benchmark   Proxy     vs SA   vs RePlAce  Overlaps
ibm01       1.041   +17.5%      -8.9%       147
ibm02       1.575   +17.4%     +14.3%       267
ibm03       1.353   +22.2%      -2.4%       268
ibm04       1.429    +5.0%      -9.7%       285
ibm06       1.660   +33.8%      -2.5%       173
ibm07       1.718   +15.1%     -17.4%       288
ibm08       1.550   +19.5%      -8.5%       303
ibm09       1.348    +2.9%     -20.4%       281
ibm10       1.583   +25.0%      -5.5%      1246
ibm11       1.275   +25.5%      -8.3%       398
ibm12       1.770   +37.4%      -2.6%       728
ibm13       1.459   +23.8%      -9.3%       476
ibm14       1.941   +14.7%     -25.8%       687
ibm15       1.733   +24.7%     -14.3%       543
ibm16       1.917   +14.2%     -29.7%       577
ibm17       2.183   +40.6%     -32.7%      1074
ibm18       2.314   +16.6%     -30.6%       406
AVG         1.641   +22.8%     -12.6%      8147
```
Status: DISQUALIFIED (8147 total overlaps). Proxy scores beat RePlAce on average by 12.6%.

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

### NOT yet implemented
- Legalization (overlap removal via spiral search)
- Nesterov momentum + BB step size
- Genetic algorithm over macro orientations
