"""
quadratic_placer.py — Quadratic WL initializer (mIP phase of ePlace-MS).

Solves L_mm * x = b (CLIQUE net model) for a low-wirelength initial placement,
then refines with B2B (Bound-to-Bound) iterations to better approximate HPWL.

Net models
----------
CLIQUE  — fully connect all k pins in a net with weight 2/(k*(k-1)).
          One-shot solve; O(k²) springs per net.
B2B     — connect each pin to the net's leftmost and rightmost pins
          (per dimension) with weight 1/(span + ε).  O(k) springs per net.
          Requires iteration as boundary pins change with positions.
Star    — for large nets (k > threshold): connect every movable pin to the
          centroid of the fixed pins in the same net.  Falls back to skip
          if no fixed pins exist.

The spring equation for a pair (pin a on macro i, pin b on macro j):
    energy = w * (xi + oa - xj - ob)²
    ∂/∂xi  → L[i,i] += w,  L[i,j] -= w,  b[i] += w*(ob - oa)
    ∂/∂xj  → L[j,j] += w,  L[j,i] -= w,  b[j] += w*(oa - ob)
If j is fixed at effective position fp_j:
    L[i,i] += w,  b[i] += w*(fp_j - oa)
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch


def quadratic_init(net_data, benchmark, cfg):
    """
    Quadratic WL minimizer for initial placement (mIP).

    Parameters
    ----------
    net_data  : dict returned by _buildNetData
    benchmark : Benchmark object
    cfg       : params sub-dict from config.toml

    Returns
    -------
    pos : FloatTensor [num_macros, 2]  — initial positions clamped to canvas.
    """
    b2b_iters        = cfg.get("quad_b2b_iters", 3)
    net_size_thresh  = cfg.get("quad_net_size_threshold", 50)

    num_macros = benchmark.num_macros
    canvas_w   = float(benchmark.canvas_width)
    canvas_h   = float(benchmark.canvas_height)
    half_w     = benchmark.macro_sizes[:num_macros, 0].numpy() / 2  # [M]
    half_h     = benchmark.macro_sizes[:num_macros, 1].numpy() / 2  # [M]

    fixed_mask_np = benchmark.macro_fixed[:num_macros].numpy()      # [M] bool
    movable_ids   = np.where(~fixed_mask_np)[0]
    n_mov         = len(movable_ids)

    if n_mov == 0:
        # All macros fixed — nothing to do.
        return benchmark.macro_positions[:num_macros].clone().float()

    # global macro index → local movable index (−1 if fixed)
    g2l = np.full(num_macros, -1, dtype=np.int64)
    g2l[movable_ids] = np.arange(n_mov)

    # macro centers [M, 2] — used to compute effective positions of fixed-macro pins
    macro_centers = benchmark.macro_positions[:num_macros].numpy().astype(np.float64)

    # ── Unpack net_data ──────────────────────────────────────────────────────
    macro_ids_np = net_data["macro_ids"].numpy()   # [P]  −1 = port
    offsets_np   = net_data["offsets"].numpy()      # [P, 2]
    net_ids_np   = net_data["net_ids"].numpy()      # [P]
    is_macro_np  = net_data["is_macro"].numpy()     # [P] bool
    num_nets     = net_data["num_nets"]

    # ── Per-pin classification ───────────────────────────────────────────────
    # Safe index for numpy indexing (ports remapped to 0, then masked out)
    pin_safe = np.where(~is_macro_np, 0, macro_ids_np)

    # A pin is "fixed" if it is a port (is_macro=False) OR its macro is fixed.
    pin_is_fixed = ~is_macro_np | fixed_mask_np[pin_safe]

    # Local movable index per pin (−1 if fixed)
    pin_local = np.where(pin_is_fixed, np.int64(-1), g2l[pin_safe])

    # Effective fixed position per pin:
    #   port      → offsets (already absolute)
    #   fixed macro → macro_center + offset
    pin_fixed_eff = np.where(
        (~is_macro_np)[:, None],
        offsets_np,
        macro_centers[pin_safe] + offsets_np,
    )  # [P, 2]

    # ── Sort pins by net for O(1) per-net slicing ────────────────────────────
    order     = np.argsort(net_ids_np, kind="stable")
    s_net     = net_ids_np[order]
    s_loc     = pin_local[order]       # local movable idx or −1
    s_fix     = pin_is_fixed[order]    # bool
    s_fpos    = pin_fixed_eff[order]   # [P, 2] effective fixed pos
    s_ofs     = offsets_np[order]      # [P, 2] offset from macro center

    net_starts = np.searchsorted(s_net, np.arange(num_nets), side="left")
    net_ends   = np.searchsorted(s_net, np.arange(num_nets), side="right")

    # ── Initial guess: canvas centre ─────────────────────────────────────────
    pos = np.zeros((n_mov, 2), dtype=np.float64)
    pos[:, 0] = canvas_w / 2.0
    pos[:, 1] = canvas_h / 2.0

    # ── CLIQUE warm-start solve ───────────────────────────────────────────────
    pos = _solve(pos, n_mov, num_nets, net_starts, net_ends,
                 s_loc, s_fix, s_fpos, s_ofs, net_size_thresh, b2b_pos=None)
    _clamp(pos, movable_ids, half_w, half_h, canvas_w, canvas_h)

    # ── B2B refinement ────────────────────────────────────────────────────────
    for _ in range(b2b_iters):
        pos = _solve(pos, n_mov, num_nets, net_starts, net_ends,
                     s_loc, s_fix, s_fpos, s_ofs, net_size_thresh, b2b_pos=pos)
        _clamp(pos, movable_ids, half_w, half_h, canvas_w, canvas_h)

    # ── Scatter back into full [num_macros, 2] tensor ─────────────────────────
    result = benchmark.macro_positions[:num_macros].clone().float()
    result[movable_ids] = torch.from_numpy(pos.astype(np.float32))
    hw = torch.from_numpy(half_w.astype(np.float32))
    hh = torch.from_numpy(half_h.astype(np.float32))
    result[:, 0] = result[:, 0].clamp(hw, canvas_w - hw)
    result[:, 1] = result[:, 1].clamp(hh, canvas_h - hh)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(pos, movable_ids, half_w, half_h, canvas_w, canvas_h):
    """Clamp movable-macro positions in-place."""
    hw = half_w[movable_ids]
    hh = half_h[movable_ids]
    np.clip(pos[:, 0], hw, canvas_w - hw, out=pos[:, 0])
    np.clip(pos[:, 1], hh, canvas_h - hh, out=pos[:, 1])


def _solve(current_pos, n_mov, num_nets, net_starts, net_ends,
           s_loc, s_fix, s_fpos, s_ofs, net_size_thresh, b2b_pos):
    """
    Build and solve L_mm * x = b for one CLIQUE or B2B pass.

    b2b_pos=None  → CLIQUE model (one-shot)
    b2b_pos=pos   → B2B model (uses current positions to find boundary pins)

    Returns new positions [n_mov, 2] as float64 ndarray.
    """
    rows_x, cols_x, vals_x = [], [], []
    rows_y, cols_y, vals_y = [], [], []
    b_x = np.zeros(n_mov, dtype=np.float64)
    b_y = np.zeros(n_mov, dtype=np.float64)

    for k in range(num_nets):
        st, en = net_starts[k], net_ends[k]
        kp = en - st
        if kp < 2:
            continue

        loc   = s_loc[st:en]   # [kp] int64, −1 if fixed
        fix   = s_fix[st:en]   # [kp] bool
        fpos  = s_fpos[st:en]  # [kp, 2]
        ofs   = s_ofs[st:en]   # [kp, 2]

        if b2b_pos is None:
            # ── CLIQUE ──────────────────────────────────────────────────────
            if kp > net_size_thresh:
                _add_star_net(loc, fix, fpos, ofs, kp,
                              rows_x, cols_x, vals_x, b_x,
                              rows_y, cols_y, vals_y, b_y)
            else:
                w = 2.0 / (kp * (kp - 1))
                _add_clique_net(loc, fix, fpos, ofs, w,
                                rows_x, cols_x, vals_x, b_x,
                                rows_y, cols_y, vals_y, b_y)
        else:
            # ── B2B ─────────────────────────────────────────────────────────
            # Compute effective positions of all pins in this net.
            eff = np.empty((kp, 2), dtype=np.float64)
            for pi in range(kp):
                if fix[pi]:
                    eff[pi] = fpos[pi]
                else:
                    eff[pi] = b2b_pos[loc[pi]] + ofs[pi]
            _add_b2b_net(loc, fix, fpos, ofs, eff,
                         rows_x, cols_x, vals_x, b_x,
                         rows_y, cols_y, vals_y, b_y)

    if not rows_x:
        return current_pos.copy()

    eps = 1e-5
    I = scipy.sparse.eye(n_mov, format="csr") * eps
    Lx = scipy.sparse.csr_matrix(
        (vals_x, (rows_x, cols_x)), shape=(n_mov, n_mov)) + I
    Ly = scipy.sparse.csr_matrix(
        (vals_y, (rows_y, cols_y)), shape=(n_mov, n_mov)) + I

    x_new = scipy.sparse.linalg.spsolve(Lx, b_x)
    y_new = scipy.sparse.linalg.spsolve(Ly, b_y)

    pos_new = np.stack([x_new, y_new], axis=1)
    bad = ~np.isfinite(pos_new).all(axis=1)
    pos_new[bad] = current_pos[bad]
    return pos_new


def _add_clique_net(loc, fix, fpos, ofs, w,
                    rows_x, cols_x, vals_x, b_x,
                    rows_y, cols_y, vals_y, b_y):
    """Add CLIQUE springs (weight w per pair) for one net."""
    k = len(loc)
    for a in range(k):
        for b in range(a + 1, k):
            _add_spring(loc[a], loc[b], fix[a], fix[b],
                        ofs[a, 0], ofs[b, 0], fpos[a, 0], fpos[b, 0], w,
                        rows_x, cols_x, vals_x, b_x)
            _add_spring(loc[a], loc[b], fix[a], fix[b],
                        ofs[a, 1], ofs[b, 1], fpos[a, 1], fpos[b, 1], w,
                        rows_y, cols_y, vals_y, b_y)


def _add_star_net(loc, fix, fpos, ofs, kp,
                  rows_x, cols_x, vals_x, b_x,
                  rows_y, cols_y, vals_y, b_y):
    """
    Star model for large nets: connect each movable pin to the centroid of
    all fixed pins in the net.  Skips if no fixed pins exist.
    """
    fixed_idx   = np.where(fix)[0]
    movable_idx = np.where(~fix)[0]
    if len(fixed_idx) == 0 or len(movable_idx) == 0:
        return
    cx = fpos[fixed_idx, 0].mean()
    cy = fpos[fixed_idx, 1].mean()
    # Use the same per-pair weight as CLIQUE would for a consistent magnitude.
    w = 2.0 / (kp * (kp - 1))
    for mi in movable_idx:
        i = loc[mi]
        rows_x.append(i); cols_x.append(i); vals_x.append(w)
        b_x[i] += w * (cx - ofs[mi, 0])
        rows_y.append(i); cols_y.append(i); vals_y.append(w)
        b_y[i] += w * (cy - ofs[mi, 1])


def _add_b2b_net(loc, fix, fpos, ofs, eff,
                 rows_x, cols_x, vals_x, b_x,
                 rows_y, cols_y, vals_y, b_y):
    """
    B2B model for one net.  For each dimension, find the boundary pins (min/max
    effective position) and connect every other pin to those boundaries.
    Weight = 1 / (span + ε).
    """
    eps = 1e-4
    for dim, (rows, cols, vals, b) in enumerate((
        (rows_x, cols_x, vals_x, b_x),
        (rows_y, cols_y, vals_y, b_y),
    )):
        u     = eff[:, dim]
        span  = u.max() - u.min()
        w     = 1.0 / (span + eps)
        min_p = int(np.argmin(u))
        max_p = int(np.argmax(u))

        kp = len(loc)
        for pi in range(kp):
            # Connect pi → min_p (if pi is not already the min)
            if pi != min_p:
                _add_spring(loc[pi], loc[min_p], fix[pi], fix[min_p],
                            ofs[pi, dim], ofs[min_p, dim],
                            fpos[pi, dim], fpos[min_p, dim], w,
                            rows, cols, vals, b)
            # Connect pi → max_p (if pi is not already the max)
            if pi != max_p:
                _add_spring(loc[pi], loc[max_p], fix[pi], fix[max_p],
                            ofs[pi, dim], ofs[max_p, dim],
                            fpos[pi, dim], fpos[max_p, dim], w,
                            rows, cols, vals, b)


def _add_spring(ia, ib, fa, fb, oa, ob, fpa, fpb, w, rows, cols, vals, b):
    """
    Add one spring between pin a (macro ia, offset oa) and pin b (macro ib, offset ob).
    fa / fb : True if the pin is fixed.
    fpa/fpb : effective fixed position (ignored when pin is movable).

    Spring energy: w * (xa + oa − xb − ob)²

    Movable–movable:
        L[ia,ia] += w,  L[ib,ib] += w
        L[ia,ib] -= w,  L[ib,ia] -= w
        b[ia]    += w*(ob − oa)
        b[ib]    += w*(oa − ob)

    Movable a, fixed b at fpb:
        L[ia,ia] += w,  b[ia] += w*(fpb − oa)

    Fixed a at fpa, movable b:
        L[ib,ib] += w,  b[ib] += w*(fpa − ob)

    Fixed–fixed: no contribution.
    """
    if not fa and not fb:
        rows += [ia, ib, ia, ib]
        cols += [ia, ib, ib, ia]
        vals += [w,  w, -w, -w]
        b[ia] += w * (ob - oa)
        b[ib] += w * (oa - ob)
    elif not fa:                      # a movable, b fixed
        rows.append(ia); cols.append(ia); vals.append(w)
        b[ia] += w * (fpb - oa)
    elif not fb:                      # a fixed, b movable
        rows.append(ib); cols.append(ib); vals.append(w)
        b[ib] += w * (fpa - ob)
    # both fixed → no contribution
