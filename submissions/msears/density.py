"""
density.py — Density spreading forces for MSPlacer.

Two methods, selected by density_method in config.toml:

  "electrostatic" — ePlace electrostatics (Lu et al. DAC 2015, DREAMplace Lin et al.
                    DAC 2019). Solves the Poisson equation via 2D DCT to obtain a
                    global electric field. Each macro feels the field at its location.
                    Long-range, physically motivated, better spreading.

  "bell"          — Bell-shape quadratic kernel (pre-ePlace, DREAMplace reference [14]).
                    Smooth differentiable approximation to overlap. Local range.
                    Gradient via PyTorch autograd.

Public API (used by placer.py):
    compute_density_gradient(method, pos, benchmark, target_density)
        -> (grad [n, 2] FloatTensor, energy float)
"""

import math
import torch
import torch.nn.functional as F


# ===========================================================================
# Electrostatic method
# Modelled on DREAMplace (Lin et al. DAC 2019):
#   dreamplace/ops/electric_potential/electric_potential.py
#   dreamplace/ops/dct/discrete_spectral_transform.py
#   dreamplace/ops/electric_potential/electric_overflow.py
# ===========================================================================

# ---------------------------------------------------------------------------
# DCT / DST helpers — 2N-padding trick
#
# Why DCT and not standard FFT?
#   The Poisson equation ∇²φ = ρ with Neumann boundary conditions
#   (∂φ/∂n = 0 at canvas edges — charge cannot leave) has cosines as
#   eigenfunctions. Using DCT2 (cosine basis) directly encodes these BCs,
#   giving more physically correct placement near the canvas boundary.
#
# The 2N-padding trick converts DCT2 into a standard real FFT:
#   y[u] = Σ_i x[i] * cos(π*(2i+1)*u/(2N))
#   1. Pad x with N zeros → length 2N
#   2. rfft → take first N complex values
#   3. Multiply by 2*exp(-iπk/(2N))   (precomputed as expk)
#   4. Real part → DCT coefficients
# ---------------------------------------------------------------------------

def _precompute_expk(N, dtype, device):
    """
    Precompute 2*exp(-iπk/(2N)) as a real [N, 2] tensor (cos, -sin columns).
    Used to twiddle rfft output into DCT2 coefficients.
    """
    k = torch.arange(N, dtype=dtype, device=device)
    theta = k * (math.pi / (2 * N))
    return torch.stack([theta.cos(), -theta.sin()], dim=1).mul_(2)   # [N, 2]


def _dct_2N(x, expk):
    """
    1D DCT2 via 2N-padding trick.  x: [..., N] → y: [..., N]
    Follows DREAMplace dct_2N().
    """
    N = x.shape[-1]
    x_pad = F.pad(x, (0, N))                                          # [..., 2N]
    # rfft gives [..., N+1] complex; take first N bins
    y = torch.fft.rfft(x_pad, dim=-1)[..., :N]                        # [..., N] complex
    y_real = y.real * expk[:, 0] - y.imag * expk[:, 1]                # [..., N]
    return y_real / N


def _idct_2N(x, expk):
    """
    1D IDCT2 via 2N-padding trick.  x: [..., N] → y: [..., N]

    Evaluates:  y[j] = Σ_k x[k] * cos(πk*(2j+1)/(2N))

    Twiddle: 2*exp(-iπk/(2N)).  Take irfft samples [1:N+1] (not [0:N])
    so the spatial argument is (2j+1)/(2N) rather than (2j-1)/(2N).
    """
    N = x.shape[-1]
    x_c = torch.complex(x * expk[:, 0], x * expk[:, 1])              # 2x·exp(-iπk/(2N))
    x_pad = F.pad(x_c, (0, N))                                        # [..., 2N] complex
    y = torch.fft.irfft(x_pad, n=2 * N, dim=-1)[..., 1:N + 1]        # [..., N]
    return y * N


def _idxst_2N(x, expk):
    """
    1D DST-III via 2N-padding trick.  x: [..., N] → y: [..., N]

    Evaluates:  y[j] = Σ_k x[k] * sin(πk*(2j+1)/(2N))

    This is the transform needed for the spatial derivative of the DCT2 basis:
        d/dx cos(πk*(2j+1)/(2N)) ∝ -k * sin(πk*(2j+1)/(2N))

    Output is antisymmetric around the density peak (negative on the right,
    positive on the left), which gives the correct repulsive spreading force
    when used as den_grad in  pos -= λ * den_grad.

    Twiddle: -2i·exp(-iπk/(2N))
        real = -2·sin(πk/(2N)) =  expk[:,1]
        imag = -2·cos(πk/(2N)) = -expk[:,0]

    Bug fixed: the old version used expkp1 = 2*exp(+iπ(k+1)/(2N)) which
    produced a cosine output (always positive) instead of a sine.
    """
    N = x.shape[-1]
    x_c = torch.complex(x * expk[:, 1], -x * expk[:, 0])             # -2ix·exp(-iπk/(2N))
    x_pad = F.pad(x_c, (0, N))                                        # [..., 2N]
    y = torch.fft.irfft(x_pad, n=2 * N, dim=-1)[..., 1:N + 1]        # [..., N]
    return y * N


# 2D composite transforms
# Each operates on a [rows, cols] tensor.
# Convention: dim 0 = rows (y-axis), dim 1 = cols (x-axis).

def _dct2_2d(x, expk_rows, expk_cols):
    """2D DCT2: DCT along cols (x), then DCT along rows (y)."""
    return _dct_2N(_dct_2N(x, expk_cols).T, expk_rows).T


def _idxst_idct_2d(x, expk_rows, expk_cols):
    """
    IDCT along rows (y), then IDXST (DST-III) along cols (x).
    → x-component of E field: antisymmetric in x, symmetric in y.
    """
    return _idxst_2N(_idct_2N(x.T, expk_rows).T, expk_cols)


def _idct_idxst_2d(x, expk_rows, expk_cols):
    """
    IDXST (DST-III) along rows (y), then IDCT along cols (x).
    → y-component of E field: symmetric in x, antisymmetric in y.
    """
    return _idct_2N(_idxst_2N(x.T, expk_rows).T, expk_cols)


# ---------------------------------------------------------------------------
# Cache for precomputed spectral constants (keyed by grid shape + dtype)
# ---------------------------------------------------------------------------
_spectral_cache: dict = {}


def _get_spectral_constants(rows, cols, dtype, device):
    """
    Return (expk_rows, expk_cols, wu_by_ww, wv_by_ww) precomputed for a
    given grid.  Results are cached so they are computed only once per
    unique (rows, cols, dtype, device) combination.

    expk_rows/cols: twiddle factors for both DCT2 (forward) and the
                    IDCT2/DST-III inverse transforms — all use the same
                    2*exp(-iπk/(2N)) twiddle now that the IDXST bug is fixed.
    wu_by_ww: row-frequency eigenvalue filter wu/(wu²+wv²)/2  [rows, cols]
    wv_by_ww: col-frequency eigenvalue filter wv/(wu²+wv²)/2  [rows, cols]
    """
    key = (rows, cols, dtype, str(device))
    if key in _spectral_cache:
        return _spectral_cache[key]

    expk_rows = _precompute_expk(rows, dtype, device)    # [rows, 2]
    expk_cols = _precompute_expk(cols, dtype, device)    # [cols, 2]

    # Eigenvalues of the discrete Laplacian in DCT2 basis:
    #   wu[u] = 2π*u/rows  (row / y-direction frequency)
    #   wv[v] = 2π*v/cols  (col / x-direction frequency)
    wu = torch.arange(rows, dtype=dtype, device=device).mul(2 * math.pi / rows).view(rows, 1)
    wv = torch.arange(cols, dtype=dtype, device=device).mul(2 * math.pi / cols).view(1, cols)
    ww = wu ** 2 + wv ** 2
    ww[0, 0] = 1.0                           # avoid div-by-zero for DC term
    wu_by_ww = wu / ww / 2                   # [rows, cols]  y-freq filter
    wv_by_ww = wv / ww / 2                   # [rows, cols]  x-freq filter
    wu_by_ww[0, 0] = 0.0                     # zero out DC (arbitrary constant)
    wv_by_ww[0, 0] = 0.0

    result = (expk_rows, expk_cols, wu_by_ww, wv_by_ww)
    _spectral_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Exact density map builder
# ---------------------------------------------------------------------------

def _build_density_map_exact(pos, benchmark):
    """
    Build the density map [rows, cols] using exact rectangular overlap.

    Each macro's contribution to bin (r, c) is the fraction of that bin's
    area covered by the macro (after clamping):

        density[r, c] = Σ_m ratio[m] * ov_x[m, c] * ov_y[m, r] / (bw * bh)

    Clamping (from DREAMplace electric_overflow.py):
        Each macro is stretched to max(sqrt(2)*bin_size, node_size) to ensure
        it always overlaps at least one bin, preventing force discontinuities.
        The density is scaled by ratio = original_area / clamped_area to
        conserve the total placed area.

    No gradients are tracked (called inside torch.no_grad()).

    Args:
        pos       : [n, 2] macro centers (detached)
        benchmark : Benchmark object

    Returns: density_map [rows, cols]
    """
    n    = pos.shape[0]
    cw   = float(benchmark.canvas_width)
    ch   = float(benchmark.canvas_height)
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    bw   = cw / cols
    bh   = ch / rows

    sizes = benchmark.macro_sizes[:n].to(pos.dtype)    # [n, 2]
    sw    = sizes[:, 0]                                 # [n] original widths
    sh    = sizes[:, 1]                                 # [n] original heights

    sqrt2 = math.sqrt(2)
    w_cl  = sw.clamp(min=bw * sqrt2)                   # clamped width  [n]
    h_cl  = sh.clamp(min=bh * sqrt2)                   # clamped height [n]
    ratio = (sw * sh) / (w_cl * h_cl)                  # area scale [n]

    # Clamped macro extents
    x0 = pos[:, 0] - w_cl / 2                          # [n]
    x1 = pos[:, 0] + w_cl / 2
    y0 = pos[:, 1] - h_cl / 2
    y1 = pos[:, 1] + h_cl / 2

    # Bin edges along x: [cols+1], along y: [rows+1]
    bx_edges = torch.linspace(0.0, cw, cols + 1, dtype=pos.dtype)
    by_edges = torch.linspace(0.0, ch, rows + 1, dtype=pos.dtype)

    # Overlap in x: [n, cols]
    ov_x = (torch.min(x1.unsqueeze(1), bx_edges[1:].unsqueeze(0))
          - torch.max(x0.unsqueeze(1), bx_edges[:-1].unsqueeze(0))).clamp(min=0.0)
    # Overlap in y: [n, rows]
    ov_y = (torch.min(y1.unsqueeze(1), by_edges[1:].unsqueeze(0))
          - torch.max(y0.unsqueeze(1), by_edges[:-1].unsqueeze(0))).clamp(min=0.0)

    # density[r, c] = Σ_m ratio[m] * ov_y[m,r] * ov_x[m,c] / (bw*bh)
    density_map = (ov_y * ratio.unsqueeze(1)).T @ ov_x / (bw * bh)    # [rows, cols]
    return density_map


# ---------------------------------------------------------------------------
# Poisson FFT solver
# ---------------------------------------------------------------------------

def _poisson_fft_solve(density_map, bw, bh,
                        expk_rows, expk_cols,
                        wu_by_ww, wv_by_ww):
    """
    Solve ∇²φ = ρ via 2D DCT, following DREAMplace electric_potential.py.

    Steps:
      1. Normalise:   rho = density_map / (bw * bh)
      2. 2D DCT2:     auv = DCT2(rho)                            [rows, cols]
      3. x-filter:    auv_x = auv * wv_by_ww  (col/x frequency)
      4. y-filter:    auv_y = auv * wu_by_ww  (row/y frequency)
      5. Ex = IDCT_y(DST-III_x(auv_x))   antisymmetric in x, symmetric in y
      6. Ey = DST-III_y(IDCT_x(auv_y))   symmetric in x, antisymmetric in y

    The DST-III (_idxst_2N) computes  Σ_k a[k]*sin(πk*(2j+1)/(2N)),
    which is the correct spatial derivative of the cosine basis and
    produces an antisymmetric field that pushes macros away from
    high-density regions when used as  pos -= λ * (Ex, Ey).

    Returns:
        Ex   [rows, cols]  x-component of electric field (positive LEFT of source)
        Ey   [rows, cols]  y-component of electric field (positive BELOW source)
    """
    rho = density_map / (bw * bh)
    auv = _dct2_2d(rho, expk_rows, expk_cols)

    # For Ex: apply x-frequency filter (col direction = wv), then
    #         IDCT along rows (y) and DST-III along cols (x).
    # For Ey: apply y-frequency filter (row direction = wu), then
    #         DST-III along rows (y) and IDCT along cols (x).
    auv_x = auv * wv_by_ww   # col/x-frequency
    auv_y = auv * wu_by_ww   # row/y-frequency

    # TODO: should Ex use wu and Ey use wv?

    Ex = _idxst_idct_2d(auv_x, expk_rows, expk_cols)
    Ey = _idct_idxst_2d(auv_y, expk_rows, expk_cols)

    return Ex, Ey


# ---------------------------------------------------------------------------
# Bilinear interpolation of a 2D field at continuous positions
# ---------------------------------------------------------------------------

def _bilinear_interp(field, x, y, bw, bh, rows, cols):
    """
    Bilinear interpolation of grid field [rows, cols] at positions (x[n], y[n]).

    Converts canvas coordinates to fractional bin indices, then interpolates
    between the four nearest bin centers.

    Args:
        field : [rows, cols]
        x, y  : [n] macro center coordinates
        bw, bh: bin width, height
        rows, cols: grid dimensions

    Returns: interpolated values [n]
    """
    # Fractional bin index (bin centers are at bw/2, 3bw/2, ...)
    ix = (x / bw - 0.5).clamp(0.0, cols - 1.0 - 1e-6)   # [n]
    iy = (y / bh - 0.5).clamp(0.0, rows - 1.0 - 1e-6)   # [n]

    i0 = ix.long()
    j0 = iy.long()
    fx = ix - i0.float()    # fractional part in x [n]
    fy = iy - j0.float()    # fractional part in y [n]
    i1 = (i0 + 1).clamp(max=cols - 1)
    j1 = (j0 + 1).clamp(max=rows - 1)

    return ((1 - fx) * (1 - fy) * field[j0, i0]
          +      fx  * (1 - fy) * field[j0, i1]
          + (1 - fx) *      fy  * field[j1, i0]
          +      fx  *      fy  * field[j1, i1])


# ---------------------------------------------------------------------------
# Electrostatic density gradient
# ---------------------------------------------------------------------------

def _density_gradient_electrostatic(pos, benchmark, target_density):
    """
    Compute electrostatic density gradient and energy.

    Data flow (all inside torch.no_grad()):
      pos  →  exact density map  →  Poisson FFT  →  (Ex, Ey) fields
           →  bilinear interp at macro centers  →  per-macro force [n, 2]

    Args:
        pos            : [n, 2] macro centers
        benchmark      : Benchmark object
        target_density : float

    Returns:
        grad           [n, 2] FloatTensor  (force pointing away from dense regions)
        energy         float               (electrostatic energy, for logging)
        overflow_ratio float               normalised overflow (DREAMplace convention)
        max_density    float               max bin density
    """
    n    = pos.shape[0]
    cw   = float(benchmark.canvas_width)
    ch   = float(benchmark.canvas_height)
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    bw   = cw / cols
    bh   = ch / rows

    (expk_rows, expk_cols,
     wu_by_ww, wv_by_ww) = _get_spectral_constants(rows, cols, pos.dtype, pos.device)

    with torch.no_grad():
        density_map = _build_density_map_exact(pos, benchmark)

        Ex, Ey = _poisson_fft_solve(
            density_map, bw, bh,
            expk_rows, expk_cols,
            wu_by_ww, wv_by_ww)

        # Interpolate field at each macro center.
        # Sign: φ has a maximum at the density peak, so E = IDXST(auv) points
        # INTO the peak (attractive). The spreading gradient is +∇φ = -E,
        # which pushes macros away from high-density regions.
        # With pos -= λ * den_grad, den_grad = -E gives repulsion.
        fx = -_bilinear_interp(Ex, pos[:, 0], pos[:, 1], bw, bh, rows, cols)  # [n]
        fy = -_bilinear_interp(Ey, pos[:, 0], pos[:, 1], bw, bh, rows, cols)  # [n]

        grad = torch.stack([fx, fy], dim=1)   # [n, 2]

        # Electrostatic energy: ½ Σ_b ρ_b * (ρ_b - ρ_target) * bw * bh
        energy = 0.5 * float(
            (density_map * (density_map - target_density)).sum() * bw * bh)

        # Overflow metrics (DREAMplace electric_overflow.py convention)
        #   overflow_ratio = Σ max(0, ρ - target) / num_bins
        #   max_density    = max(ρ) across all bins
        ovf = (density_map - target_density).clamp(min=0.0)
        overflow_ratio = ovf.sum().item() / density_map.numel()
        max_density    = density_map.max().item()

    return grad, energy, overflow_ratio, max_density


# ===========================================================================
# Public dispatcher
# ===========================================================================

def compute_density_gradient(method, pos, benchmark, target_density):
    """
    Compute the density gradient (spreading force) for each macro.

    Args:
        method         : "bell" or "electrostatic"
        pos            : [n, 2] FloatTensor, macro centers (no requires_grad needed)
        benchmark      : Benchmark object (provides canvas size, grid, macro sizes)
        target_density : float — target area utilisation per grid bin (e.g. 0.8)

    Returns:
        grad           [n, 2] FloatTensor — force per macro
        energy         float              — density energy (for logging)
        overflow_ratio float              — Σ max(0,ρ-target)/num_bins (DREAMplace metric)
        max_density    float              — max(ρ) across all bins
    """
    if method == "bell":
        return _density_gradient_bell(pos, benchmark, target_density)
    elif method == "electrostatic":
        return _density_gradient_electrostatic(pos, benchmark, target_density)
    else:
        raise ValueError(
            f"Unknown density_method: {method!r}. Choose 'bell' or 'electrostatic'.")

# ===========================================================================
# Bell-shape method
# (Moved from placer.py; functions renamed to make the method explicit.)
# ===========================================================================

def _bell_density_fn_1d(d, node_size, bin_size):
    """
    Piecewise quadratic C¹ bell function (DREAMplace density kernel ref [14]).
    Returns smooth approximation to overlap between a macro and a bin in 1D.

    d         : [n, bins] absolute distance |center - bin_center|
    node_size : [n, 1]    macro width or height
    bin_size  : scalar    bin width or height

    Two cases depending on node_size vs bin_size:
      large macro (>= bin):  a = 4/(3*(w+bw)²),       b = 2/(3*(w+bw))
      small macro (< bin):   a = 4/((w+2bw)*(w+4bw)), b = 2/(w+4bw)
    """
    large = node_size >= bin_size                                          # [n, 1]
    denom_large = 3.0 * (node_size + bin_size) ** 2
    denom_small = (node_size + 2 * bin_size) * (node_size + 4 * bin_size)
    a = torch.where(large, 4.0 / denom_large, 4.0 / denom_small)         # [n, 1]
    b = torch.where(large,
                    2.0 / (3.0 * (node_size + bin_size)),
                    2.0 / (node_size + 4 * bin_size))                     # [n, 1]
    t1 = torch.where(large,
                     (node_size + bin_size) / 2,
                     (node_size + 2 * bin_size) / 2)                      # [n, 1]
    t2 = torch.where(large,
                     (node_size + 2 * bin_size) / 2,
                     (node_size + 4 * bin_size) / 2)                      # [n, 1]

    r1 = 1.0 - a * d ** 2
    r2 = b * (t2 - d) ** 2
    return torch.where(d <= t1, r1,
           torch.where(d <= t2, r2,
           torch.zeros_like(d)))


def _density_map_bell(pos, benchmark, target_density):
    """
    Smooth density penalty using the bell-shape kernel.
    Returns a scalar loss suitable for autograd differentiation.

      density[r,c] = Σ_m bell_y[m,r] * bell_x[m,c] / (bw * bh)
      loss         = Σ_{r,c} max(0, density[r,c] - target_density)²

    Args:
        pos            : [n, 2] macro centers, requires_grad=True
        benchmark      : Benchmark object
        target_density : float

    Returns: scalar loss tensor
    """
    n = pos.shape[0]
    cw   = float(benchmark.canvas_width)
    ch   = float(benchmark.canvas_height)
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    bw   = cw / cols
    bh   = ch / rows

    bx = torch.linspace(bw / 2, cw - bw / 2, cols, dtype=pos.dtype)  # [cols]
    by = torch.linspace(bh / 2, ch - bh / 2, rows, dtype=pos.dtype)  # [rows]

    macro_sizes = benchmark.macro_sizes[:n].to(pos.dtype)
    w = macro_sizes[:, 0].unsqueeze(1)   # [n, 1]
    h = macro_sizes[:, 1].unsqueeze(1)   # [n, 1]

    dx = (pos[:, 0].unsqueeze(1) - bx.unsqueeze(0)).abs()   # [n, cols]
    dy = (pos[:, 1].unsqueeze(1) - by.unsqueeze(0)).abs()   # [n, rows]

    bell_x = _bell_density_fn_1d(dx, w, bw)   # [n, cols]
    bell_y = _bell_density_fn_1d(dy, h, bh)   # [n, rows]

    density_map = (bell_y.T @ bell_x) / (bw * bh)   # [rows, cols]
    overflow    = (density_map - target_density).clamp(min=0.0)
    return overflow.pow(2).sum()


def _overflow_from_bell_map(density_map, target_density):
    """
    Compute overflow metric from a bell density map.
    Returns (overflow_ratio, max_density) following DREAMplace convention.

    overflow_ratio = sum(max(0, ρ - target)) / num_bins
        Fraction of total bin capacity that is over-target.
        DREAMplace stops global placement when this < stop_overflow (default 0.1).

    max_density = max(ρ) across all bins
        Direct upper bound on bin utilisation.
    """
    ovf = (density_map - target_density).clamp(min=0.0)
    overflow_ratio = ovf.sum().item() / density_map.numel()
    max_density    = density_map.max().item()
    return overflow_ratio, max_density


def _density_gradient_bell(pos, benchmark, target_density):
    """
    Compute density gradient, energy, and overflow using the bell-shape kernel.
    Uses torch.autograd.grad (not .backward()) to keep it self-contained.

    Returns:
        grad           [n, 2] FloatTensor
        energy         float
        overflow_ratio float  — normalised overflow (DREAMplace convention)
        max_density    float  — max bin density
    """
    pos_var = pos.detach().requires_grad_(True)
    # Recompute density map to get overflow metrics (autograd needs it anyway)
    n    = pos_var.shape[0]
    cw   = float(benchmark.canvas_width)
    ch   = float(benchmark.canvas_height)
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    bw   = cw / cols
    bh   = ch / rows
    bx   = torch.linspace(bw / 2, cw - bw / 2, cols, dtype=pos_var.dtype)
    by   = torch.linspace(bh / 2, ch - bh / 2, rows, dtype=pos_var.dtype)
    macro_sizes = benchmark.macro_sizes[:n].to(pos_var.dtype)
    w    = macro_sizes[:, 0].unsqueeze(1)
    h    = macro_sizes[:, 1].unsqueeze(1)
    dx   = (pos_var[:, 0].unsqueeze(1) - bx.unsqueeze(0)).abs()
    dy   = (pos_var[:, 1].unsqueeze(1) - by.unsqueeze(0)).abs()
    bell_x = _bell_density_fn_1d(dx, w, bw)
    bell_y = _bell_density_fn_1d(dy, h, bh)
    density_map = (bell_y.T @ bell_x) / (bw * bh)
    overflow_map = (density_map - target_density).clamp(min=0.0)
    energy = overflow_map.pow(2).sum()

    (grad,) = torch.autograd.grad(energy, pos_var)

    with torch.no_grad():
        overflow_ratio, max_density = _overflow_from_bell_map(
            density_map.detach(), target_density)

    return grad.detach(), energy.item(), overflow_ratio, max_density
