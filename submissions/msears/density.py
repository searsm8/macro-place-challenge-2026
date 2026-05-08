"""
density.py — Density spreading forces for CometPlacer.

Two methods, selected by density_method in config.toml:

  "electrostatic" — ePlace electrostatics (Lu et al. DAC 2015, DREAMplace Lin et al.
                    DAC 2019). Solves the Poisson equation via 2D DCT to obtain a
                    global electric field. Each macro feels the field at its location.

  "bell"          — Bell-shape quadratic kernel (pre-ePlace, DREAMplace reference [14]).
                    Smooth differentiable approximation to overlap. Local range.
                    Gradient via PyTorch autograd.

Public API:
    computeDensityGradient(method, pos, benchmark, target_density)
        -> (grad [n, 2], energy float, overflow_ratio float, max_density float)
"""

import math
import torch
import torch.nn.functional as F


# ===========================================================================
# DCT / DST helpers — 2N-padding trick
#
# The Poisson equation with Neumann BCs has cosines as eigenfunctions.
# DCT2 directly encodes these BCs. The 2N-padding trick converts DCT2
# into a standard real FFT:
#   1. Pad x with N zeros -> length 2N
#   2. rfft -> take first N complex values
#   3. Multiply by 2*exp(-iπk/(2N))  (precomputed as expk)
#   4. Real part -> DCT coefficients
# ===========================================================================

def _precomputeExpk(length, dtype, device):
    """Precompute 2*exp(-iπk/(2N)) as a real [N, 2] tensor (cos, -sin)."""
    k = torch.arange(length, dtype=dtype, device=device)
    theta = k * (math.pi / (2 * length))
    return torch.stack([theta.cos(), -theta.sin()], dim=1).mul_(2)


def _dct2N(x, expk):
    """1D DCT2 via 2N-padding trick. x: [..., N] -> y: [..., N]"""
    length = x.shape[-1]
    x_pad = F.pad(x, (0, length))
    y = torch.fft.rfft(x_pad, dim=-1)[..., :length]
    y_real = y.real * expk[:, 0] - y.imag * expk[:, 1]
    return y_real / length


def _idct2N(x, expk):
    """
    1D IDCT2 via 2N-padding trick. x: [..., N] -> y: [..., N]
    Evaluates: y[j] = Σ_k x[k] * cos(πk*(2j+1)/(2N))
    """
    length = x.shape[-1]
    x_c = torch.complex(x * expk[:, 0], x * expk[:, 1])
    x_pad = F.pad(x_c, (0, length))
    y = torch.fft.irfft(x_pad, n=2 * length, dim=-1)[..., 1:length + 1]
    return y * length


def _idxst2N(x, expk):
    """
    1D DST-III via 2N-padding trick. x: [..., N] -> y: [..., N]
    Evaluates: y[j] = Σ_k x[k] * sin(πk*(2j+1)/(2N))

    This is the spatial derivative of the DCT2 basis. The twiddle is
    -2i*exp(-iπk/(2N)), giving complex(expk[:,1], -expk[:,0]).
    """
    length = x.shape[-1]
    x_c = torch.complex(x * expk[:, 1], -x * expk[:, 0])
    x_pad = F.pad(x_c, (0, length))
    y = torch.fft.irfft(x_pad, n=2 * length, dim=-1)[..., 1:length + 1]
    return y * length


# ---------------------------------------------------------------------------
# 2D composite transforms
# Convention: dim 0 = rows (y-axis), dim 1 = cols (x-axis).
# ---------------------------------------------------------------------------

def _dct2_2d(x, expk_rows, expk_cols):
    """2D DCT2: DCT along cols (x), then DCT along rows (y)."""
    return _dct2N(_dct2N(x, expk_cols).T, expk_rows).T


def _idxstIdct2d(x, expk_rows, expk_cols):
    """IDCT along rows (y), then IDXST along cols (x) -> Ex field."""
    return _idxst2N(_idct2N(x.T, expk_rows).T, expk_cols)


def _idctIdxst2d(x, expk_rows, expk_cols):
    """IDXST along rows (y), then IDCT along cols (x) -> Ey field."""
    return _idct2N(_idxst2N(x.T, expk_rows).T, expk_cols)


# ---------------------------------------------------------------------------
# Spectral constants cache
# ---------------------------------------------------------------------------
_spectral_cache: dict = {}


def _getSpectralConstants(rows, cols, dtype, device):
    """
    Return (expk_rows, expk_cols, wu_by_ww, wv_by_ww) for a grid shape.
    Cached per unique (rows, cols, dtype, device).
    """
    key = (rows, cols, dtype, str(device))
    if key in _spectral_cache:
        return _spectral_cache[key]

    expk_rows = _precomputeExpk(rows, dtype, device)
    expk_cols = _precomputeExpk(cols, dtype, device)

    wu = torch.arange(rows, dtype=dtype, device=device).mul(2 * math.pi / rows).view(rows, 1)
    wv = torch.arange(cols, dtype=dtype, device=device).mul(2 * math.pi / cols).view(1, cols)
    ww = wu ** 2 + wv ** 2
    ww[0, 0] = 1.0
    wu_by_ww = wu / ww / 2
    wv_by_ww = wv / ww / 2
    wu_by_ww[0, 0] = 0.0
    wv_by_ww[0, 0] = 0.0

    result = (expk_rows, expk_cols, wu_by_ww, wv_by_ww)
    _spectral_cache[key] = result
    return result


# ===========================================================================
# Electrostatic method
# ===========================================================================

def _buildDensityMapExact(pos, benchmark, target_density, density_mask=None,
                          halo_size=0.0, grid_rows=None, grid_cols=None):
    """
    Build density map [rows, cols] using exact rectangular overlap.

    Each macro is clamped to min size sqrt(2)*bin_size to ensure it always
    overlaps at least one bin. Density is rescaled by original/clamped area
    to conserve total placed area.

    density_mask: optional bool tensor [n]. When provided, only macros where
        density_mask[i] is True contribute area to the density map. The
        returned map reflects only those macros.
    halo_size: fractional padding applied to hard macro footprints before
        density computation (0.0 = disabled). Hard macros are inflated to
        w*(1+halo_size) × h*(1+halo_size); area_ratio is adjusted so the
        total charge (placed area) in the density field is conserved.
        Applied before density_mask filtering so the hard/soft split is
        preserved regardless of which macros are masked in.
    grid_rows, grid_cols: override benchmark grid dimensions (e.g. 512×512).
    """
    num_macros = pos.shape[0]
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    rows = grid_rows if grid_rows is not None else benchmark.grid_rows
    cols = grid_cols if grid_cols is not None else benchmark.grid_cols
    bin_w = canvas_w / cols
    bin_h = canvas_h / rows

    real_sizes = benchmark.macro_sizes[:num_macros].to(device=pos.device, dtype=pos.dtype)  # [n, 2]

    # Inflate hard macro footprints by (1 + halo_size) before masking so the
    # hard/soft classification survives density_mask filtering below.
    if halo_size > 0.0:
        num_hard = benchmark.num_hard_macros
        eff_sizes = real_sizes.clone()
        eff_sizes[:num_hard] *= (1.0 + halo_size)
    else:
        eff_sizes = real_sizes

    if density_mask is not None:
        density_pos = pos[density_mask]
        real_sizes  = real_sizes[density_mask]
        eff_sizes   = eff_sizes[density_mask]
    else:
        density_pos = pos

    orig_w = real_sizes[:, 0]   # actual macro dimensions (charge numerator)
    orig_h = real_sizes[:, 1]
    eff_w  = eff_sizes[:, 0]    # halo-inflated dimensions (footprint denominator)
    eff_h  = eff_sizes[:, 1]

    sqrt2 = math.sqrt(2)
    clamp_w = eff_w.clamp(min=bin_w * sqrt2)
    clamp_h = eff_h.clamp(min=bin_h * sqrt2)
    area_ratio = (orig_w * orig_h) / (clamp_w * clamp_h)

    # Clamped macro extents (using only the macros that contribute density)
    x_lo = density_pos[:, 0] - clamp_w / 2
    x_hi = density_pos[:, 0] + clamp_w / 2
    y_lo = density_pos[:, 1] - clamp_h / 2
    y_hi = density_pos[:, 1] + clamp_h / 2

    # Bin edges
    bx_edges = torch.linspace(0.0, canvas_w, cols + 1, dtype=pos.dtype, device=pos.device)
    by_edges = torch.linspace(0.0, canvas_h, rows + 1, dtype=pos.dtype, device=pos.device)

    # Overlap in x: [n, cols] and y: [n, rows]
    overlap_x = (torch.min(x_hi.unsqueeze(1), bx_edges[1:].unsqueeze(0))
                 - torch.max(x_lo.unsqueeze(1), bx_edges[:-1].unsqueeze(0))).clamp(min=0.0)
    overlap_y = (torch.min(y_hi.unsqueeze(1), by_edges[1:].unsqueeze(0))
                 - torch.max(y_lo.unsqueeze(1), by_edges[:-1].unsqueeze(0))).clamp(min=0.0)

    # Per-macro fractional overlaps (dimensionless): fy[m,r], fx[m,c]
    fy = (overlap_y * area_ratio.unsqueeze(1)) / bin_h   # [n, rows]
    fx = overlap_x / bin_w                                # [n, cols]

    # Cap each macro's contribution to any single bin at target_density.
    # max over (r,c) of fy[m,r]*fx[m,c] = max_r(fy[m,r]) * max_c(fx[m,c])
    # because fy and fx are nonneg and factor independently.
    max_contrib = fy.amax(dim=1) * fx.amax(dim=1)        # [n]
    scale = (target_density / max_contrib.clamp(min=1e-8)).clamp(max=1.0)  # [n]

    # density[r, c] = Σ_m scale[m] * fy[m,r] * fx[m,c]
    density_map = (fy * scale.unsqueeze(1)).T @ fx
    return density_map


def _poissonFftSolve(density_map, bin_w, bin_h,
                     expk_rows, expk_cols, wu_by_ww, wv_by_ww):
    """
    Solve ∇²φ = ρ via 2D DCT. Returns electric field components (Ex, Ey).

    Steps:
      1. Normalise rho = density_map / (bin_w * bin_h)
      2. 2D DCT2 -> spectral coefficients auv
      3. Filter by x-frequency (wv) and y-frequency (wu)
      4. Inverse transforms: IDXST+IDCT for Ex, IDCT+IDXST for Ey
    """
    rho = density_map / (bin_w * bin_h)
    auv = _dct2_2d(rho, expk_rows, expk_cols)

    auv_x = auv * wv_by_ww
    auv_y = auv * wu_by_ww

    field_ex = _idxstIdct2d(auv_x, expk_rows, expk_cols)
    field_ey = _idctIdxst2d(auv_y, expk_rows, expk_cols)

    return field_ex, field_ey


def _bilinearInterp(field, x_pos, y_pos, bin_w, bin_h, rows, cols):
    """
    Bilinear interpolation of grid field [rows, cols] at continuous
    positions (x_pos[n], y_pos[n]). Converts canvas coordinates to
    fractional bin indices, interpolates between four nearest bin centers.
    """
    ix = (x_pos / bin_w - 0.5).clamp(0.0, cols - 1.0 - 1e-6)
    iy = (y_pos / bin_h - 0.5).clamp(0.0, rows - 1.0 - 1e-6)

    i0 = ix.long()
    j0 = iy.long()
    frac_x = ix - i0.float()
    frac_y = iy - j0.float()
    i1 = (i0 + 1).clamp(max=cols - 1)
    j1 = (j0 + 1).clamp(max=rows - 1)

    return ((1 - frac_x) * (1 - frac_y) * field[j0, i0]
            + frac_x * (1 - frac_y) * field[j0, i1]
            + (1 - frac_x) * frac_y * field[j1, i0]
            + frac_x * frac_y * field[j1, i1])


def _densityGradientElectrostatic(pos, benchmark, target_density,
                                   density_mask=None, halo_size=0.0,
                                   grid_rows=None, grid_cols=None):
    """
    Compute electrostatic density gradient and metrics.

    Pipeline (all inside torch.no_grad()):
      pos -> exact density map -> Poisson FFT -> (Ex, Ey) fields
          -> bilinear interp at macro centers -> per-macro force [n, 2]

    density_mask: optional bool tensor [n]. When set, only masked macros
        contribute area to the density map (phase 2.5 hard-spread mode).
        The electric field is still interpolated at *all* macro positions so
        every macro feels the force generated by the hard-macro density field.
    halo_size: forwarded to _buildDensityMapExact — see that function.
    grid_rows, grid_cols: override benchmark grid dimensions (e.g. 512×512).

    Sign convention: φ peaks at density maxima, E = -∇φ is attractive.
    Spreading force = -E = +∇φ pushes macros away from dense regions.
    """
    num_macros = pos.shape[0]
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    rows = grid_rows if grid_rows is not None else benchmark.grid_rows
    cols = grid_cols if grid_cols is not None else benchmark.grid_cols
    bin_w = canvas_w / cols
    bin_h = canvas_h / rows

    spectral = _getSpectralConstants(rows, cols, pos.dtype, pos.device)
    expk_rows, expk_cols, wu_by_ww, wv_by_ww = spectral

    with torch.no_grad():
        density_map = _buildDensityMapExact(pos, benchmark, target_density,
                                            density_mask, halo_size,
                                            grid_rows=rows, grid_cols=cols)

        field_ex, field_ey = _poissonFftSolve(
            density_map, bin_w, bin_h,
            expk_rows, expk_cols, wu_by_ww, wv_by_ww)

        # Negate field for repulsive force (spreading away from peaks)
        force_x = -_bilinearInterp(field_ex, pos[:, 0], pos[:, 1],
                                   bin_w, bin_h, rows, cols)
        force_y = -_bilinearInterp(field_ey, pos[:, 0], pos[:, 1],
                                   bin_w, bin_h, rows, cols)

        grad = torch.stack([force_x, force_y], dim=1)

        # Electrostatic energy: ½ Σ_b ρ_b * (ρ_b - target) * bin_w * bin_h
        energy = 0.5 * float(
            (density_map * (density_map - target_density)).sum() * bin_w * bin_h)

        # Overflow metrics
        overflow_map = (density_map - target_density).clamp(min=0.0)
        overflow_ratio = overflow_map.sum().item() / density_map.numel()
        max_density = density_map.max().item()

    return grad, energy, overflow_ratio, max_density


# ===========================================================================
# Bell-shape method
# ===========================================================================

def _bellKernel1d(dist, node_size, bin_size):
    """
    Piecewise quadratic C¹ bell function (DREAMplace density kernel ref [14]).

    dist      : [n, bins] absolute distance |center - bin_center|
    node_size : [n, 1]    macro width or height
    bin_size  : scalar     bin width or height

    Two parameterisations based on node_size vs bin_size (large/small macro).
    """
    is_large = node_size >= bin_size

    denom_large = 3.0 * (node_size + bin_size) ** 2
    denom_small = (node_size + 2 * bin_size) * (node_size + 4 * bin_size)
    coeff_a = torch.where(is_large, 4.0 / denom_large, 4.0 / denom_small)
    coeff_b = torch.where(is_large,
                          2.0 / (3.0 * (node_size + bin_size)),
                          2.0 / (node_size + 4 * bin_size))
    thresh_1 = torch.where(is_large,
                           (node_size + bin_size) / 2,
                           (node_size + 2 * bin_size) / 2)
    thresh_2 = torch.where(is_large,
                           (node_size + 2 * bin_size) / 2,
                           (node_size + 4 * bin_size) / 2)

    inner = 1.0 - coeff_a * dist ** 2
    outer = coeff_b * (thresh_2 - dist) ** 2
    return torch.where(dist <= thresh_1, inner,
                       torch.where(dist <= thresh_2, outer,
                                   torch.zeros_like(dist)))


def _densityGradientBell(pos, benchmark, target_density):
    """
    Compute density gradient using bell-shape kernel + autograd.

    Builds a smooth density map via bell kernel products, penalises bins
    exceeding target with squared overflow, and differentiates via autograd.
    """
    pos_var = pos.detach().requires_grad_(True)
    num_macros = pos_var.shape[0]
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    bin_w = canvas_w / cols
    bin_h = canvas_h / rows

    bin_cx = torch.linspace(bin_w / 2, canvas_w - bin_w / 2, cols, dtype=pos_var.dtype, device=pos_var.device)
    bin_cy = torch.linspace(bin_h / 2, canvas_h - bin_h / 2, rows, dtype=pos_var.dtype, device=pos_var.device)

    macro_sizes = benchmark.macro_sizes[:num_macros].to(device=pos_var.device, dtype=pos_var.dtype)
    width = macro_sizes[:, 0].unsqueeze(1)
    height = macro_sizes[:, 1].unsqueeze(1)

    dist_x = (pos_var[:, 0].unsqueeze(1) - bin_cx.unsqueeze(0)).abs()
    dist_y = (pos_var[:, 1].unsqueeze(1) - bin_cy.unsqueeze(0)).abs()

    bell_x = _bellKernel1d(dist_x, width, bin_w)
    bell_y = _bellKernel1d(dist_y, height, bin_h)

    density_map = (bell_y.T @ bell_x) / (bin_w * bin_h)
    overflow_map = (density_map - target_density).clamp(min=0.0)
    energy = overflow_map.pow(2).sum()

    (grad,) = torch.autograd.grad(energy, pos_var)

    with torch.no_grad():
        overflow_ratio = overflow_map.detach().sum().item() / density_map.numel()
        max_density = density_map.detach().max().item()

    return grad.detach(), energy.item(), overflow_ratio, max_density


# ===========================================================================
# Public Poisson gradient helper
# ===========================================================================

def computePoissonGradient(map_2d, pos, benchmark, grid_rows=None, grid_cols=None):
    """
    Run the Poisson FFT pipeline on an arbitrary zero-mean 2D map and return
    repulsive forces at each macro position via bilinear interpolation.

    The caller is responsible for zero-meaning the map before passing it in
    (so the Neumann BCs are well-posed). Used for congestion gradient as well
    as density gradient when a pre-built map is available.

    Args:
        map_2d     : [rows, cols] FloatTensor on the correct device
        pos        : [n, 2] macro center positions
        benchmark  : Benchmark (for canvas size)
        grid_rows, grid_cols : override benchmark grid (must match map_2d shape)

    Returns:
        grad [n, 2] — repulsive force per macro (same sign convention as
                      density gradient: pushes away from map peaks)
    """
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    rows = grid_rows if grid_rows is not None else benchmark.grid_rows
    cols = grid_cols if grid_cols is not None else benchmark.grid_cols
    bin_w = canvas_w / cols
    bin_h = canvas_h / rows

    spectral = _getSpectralConstants(rows, cols, pos.dtype, pos.device)
    expk_rows, expk_cols, wu_by_ww, wv_by_ww = spectral

    with torch.no_grad():
        field_ex, field_ey = _poissonFftSolve(
            map_2d, bin_w, bin_h,
            expk_rows, expk_cols, wu_by_ww, wv_by_ww)

        force_x = -_bilinearInterp(field_ex, pos[:, 0], pos[:, 1],
                                   bin_w, bin_h, rows, cols)
        force_y = -_bilinearInterp(field_ey, pos[:, 0], pos[:, 1],
                                   bin_w, bin_h, rows, cols)

    return torch.stack([force_x, force_y], dim=1)


# ===========================================================================
# Public dispatcher
# ===========================================================================

def computeDensityGradient(method, pos, benchmark, target_density,
                           density_mask=None, halo_size=0.0,
                           grid_rows=None, grid_cols=None):
    """
    Compute density gradient (spreading force) for each macro.

    Args:
        method         : "bell" or "electrostatic"
        pos            : [n, 2] FloatTensor, macro centers
        benchmark      : Benchmark object
        target_density : float — target per-bin utilisation
        density_mask   : optional bool tensor [n] — electrostatic only.
                         When set, only masked macros contribute to the density
                         map; all macros still feel the resulting field.
        halo_size      : fractional padding for hard macro footprints — see
                         _buildDensityMapExact. Electrostatic only.
        grid_rows,
        grid_cols      : override benchmark grid dimensions (electrostatic only).

    Returns:
        grad           [n, 2] FloatTensor — force per macro
        energy         float              — density energy (logging)
        overflow_ratio float              — normalised overflow
        max_density    float              — max bin density
    """
    if method == "bell":
        return _densityGradientBell(pos, benchmark, target_density)
    elif method == "electrostatic":
        return _densityGradientElectrostatic(pos, benchmark, target_density,
                                             density_mask, halo_size,
                                             grid_rows=grid_rows, grid_cols=grid_cols)
    else:
        raise ValueError(f"Unknown density method: {method!r}")
