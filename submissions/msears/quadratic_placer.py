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

Two-stage anchor placement (quad_anchor_fraction > 0)
------------------------------------------------------
Stage 1: Select the top anchor_fraction of movable macros by area×net_degree.
         Run a CLIQUE-only solve with only ports+fixed macros as anchors to
         position these anchor macros.  Non-anchor movables are held at canvas
         centre for this solve (they contribute spring force but no unknowns).
Stage 2: Fix the anchors at their stage-1 positions.  Run the full CLIQUE+B2B
         solve for all remaining movable macros, now with both ports and internal
         anchors as fixed infrastructure.

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
    pos         : FloatTensor [num_macros, 2]  — initial positions clamped to canvas.
    scatter_ids : np.ndarray of macro indices that were scattered (empty if none).
    """
    b2b_iters        = cfg.get("quad_b2b_iters", 3)
    net_size_thresh  = cfg.get("quad_net_size_threshold", 50)
    anchor_fraction  = cfg.get("quad_anchor_fraction", 0.0)
    scatter_fraction = cfg.get("quad_scatter_fraction", 0.0)

    num_macros = benchmark.num_macros
    canvas_w   = float(benchmark.canvas_width)
    canvas_h   = float(benchmark.canvas_height)
    half_w     = benchmark.macro_sizes[:num_macros, 0].numpy() / 2  # [M]
    half_h     = benchmark.macro_sizes[:num_macros, 1].numpy() / 2  # [M]

    fixed_mask_np = benchmark.macro_fixed[:num_macros].numpy()      # [M] bool
    movable_ids   = np.where(~fixed_mask_np)[0]
    n_mov         = len(movable_ids)

    if n_mov == 0:
        return benchmark.macro_positions[:num_macros].clone().float(), np.empty(0, dtype=np.int64)

    macro_centers = benchmark.macro_positions[:num_macros].numpy().astype(np.float64)

    # ── Unpack net_data ──────────────────────────────────────────────────────
    macro_ids_np = net_data["macro_ids"].numpy()   # [P]  −1 = port
    offsets_np   = net_data["offsets"].numpy()      # [P, 2]
    net_ids_np   = net_data["net_ids"].numpy()      # [P]
    is_macro_np  = net_data["is_macro"].numpy()     # [P] bool
    num_nets     = net_data["num_nets"]

    # ── Sort pins by net once — shared by both stages ────────────────────────
    order     = np.argsort(net_ids_np, kind="stable")
    s_net     = net_ids_np[order]
    s_mid     = macro_ids_np[order]   # macro id per sorted pin
    s_ism     = is_macro_np[order]    # is_macro per sorted pin
    s_ofs     = offsets_np[order]     # [P, 2] offset per sorted pin

    net_starts = np.searchsorted(s_net, np.arange(num_nets), side="left")
    net_ends   = np.searchsorted(s_net, np.arange(num_nets), side="right")

    # ── Scatter-and-lock: randomly place biggest hard macros, then fix them ──
    if scatter_fraction > 0:
        # Candidate pool: movable hard macros only
        hard_movable = movable_ids[movable_ids < benchmark.num_hard_macros]
        if len(hard_movable) > 1:
            # Select largest until cumulative area >= scatter_fraction * total hard-movable area
            areas = (half_w * half_h)[hard_movable]
            ranked = np.argsort(areas)[::-1]
            cutoff = np.searchsorted(np.cumsum(areas[ranked]), scatter_fraction * areas.sum(), side="left")
            n_scatter = min(int(cutoff) + 1, len(hard_movable) - 1)
            scatter_ids = hard_movable[ranked[:n_scatter]]
            remain_ids  = movable_ids[~np.isin(movable_ids, scatter_ids)]

            # Random placement within valid canvas bounds
            rng = np.random.default_rng(cfg.get("seed", 42))
            scatter_pos = np.empty((n_scatter, 2), dtype=np.float64)
            scatter_pos[:, 0] = rng.uniform(half_w[scatter_ids], canvas_w - half_w[scatter_ids])
            scatter_pos[:, 1] = rng.uniform(half_h[scatter_ids], canvas_h - half_h[scatter_ids])

            # Fix scattered macros; solve quadratic for the rest
            fixed_s2   = fixed_mask_np.copy()
            fixed_s2[scatter_ids] = True
            centers_s2 = macro_centers.copy()
            centers_s2[scatter_ids] = scatter_pos

            g2l_s2 = np.full(num_macros, -1, dtype=np.int64)
            g2l_s2[remain_ids] = np.arange(len(remain_ids))

            s_loc_s2, s_fix_s2, s_fpos_s2 = _classify_pins(
                s_mid, s_ism, s_ofs, fixed_s2, centers_s2, g2l_s2)

            pos_s2 = np.empty((len(remain_ids), 2), dtype=np.float64)
            pos_s2[:, 0] = canvas_w / 2.0
            pos_s2[:, 1] = canvas_h / 2.0
            pos_s2 = _solve(pos_s2, len(remain_ids), num_nets, net_starts, net_ends,
                            s_loc_s2, s_fix_s2, s_fpos_s2, s_ofs, net_size_thresh, b2b_pos=None)
            _clamp(pos_s2, remain_ids, half_w, half_h, canvas_w, canvas_h)

            for _ in range(b2b_iters):
                pos_s2 = _solve(pos_s2, len(remain_ids), num_nets, net_starts, net_ends,
                                s_loc_s2, s_fix_s2, s_fpos_s2, s_ofs, net_size_thresh, b2b_pos=pos_s2)
                _clamp(pos_s2, remain_ids, half_w, half_h, canvas_w, canvas_h)

            return _scatter_result(benchmark, num_macros, half_w, half_h, canvas_w, canvas_h,
                                   {tuple(scatter_ids): scatter_pos, tuple(remain_ids): pos_s2}), scatter_ids

    # ── Two-stage anchor placement ───────────────────────────────────────────
    if anchor_fraction > 0:
        anchor_ids = _select_anchors(
            movable_ids, macro_ids_np, is_macro_np, net_ids_np,
            half_w, half_h, anchor_fraction)

        if len(anchor_ids) > 0:
            remain_ids = movable_ids[~np.isin(movable_ids, anchor_ids)]

            # Stage 1: solve for anchor positions (ports only as fixed infra).
            # Non-anchor movables are held at canvas centre — they act as weak
            # fixed springs pulling anchors inward, but have no unknowns.
            fixed_s1    = fixed_mask_np.copy()
            fixed_s1[anchor_ids] = False
            fixed_s1[remain_ids] = True   # non-anchor movables are fixed at canvas centre
            centers_s1  = macro_centers.copy()
            centers_s1[remain_ids, 0] = canvas_w / 2.0
            centers_s1[remain_ids, 1] = canvas_h / 2.0

            g2l_s1 = np.full(num_macros, -1, dtype=np.int64)
            g2l_s1[anchor_ids] = np.arange(len(anchor_ids))

            s_loc_s1, s_fix_s1, s_fpos_s1 = _classify_pins(
                s_mid, s_ism, s_ofs, fixed_s1, centers_s1, g2l_s1)

            pos_s1 = np.empty((len(anchor_ids), 2), dtype=np.float64)
            pos_s1[:, 0] = canvas_w / 2.0
            pos_s1[:, 1] = canvas_h / 2.0
            pos_s1 = _solve(pos_s1, len(anchor_ids), num_nets, net_starts, net_ends,
                            s_loc_s1, s_fix_s1, s_fpos_s1, s_ofs, net_size_thresh, b2b_pos=None)
            _clamp(pos_s1, anchor_ids, half_w, half_h, canvas_w, canvas_h)

            # Stage 2: fix anchors at stage-1 positions; solve for remaining macros.
            fixed_s2   = fixed_mask_np.copy()
            fixed_s2[anchor_ids] = True
            centers_s2 = macro_centers.copy()
            centers_s2[anchor_ids] = pos_s1

            g2l_s2 = np.full(num_macros, -1, dtype=np.int64)
            g2l_s2[remain_ids] = np.arange(len(remain_ids))

            s_loc_s2, s_fix_s2, s_fpos_s2 = _classify_pins(
                s_mid, s_ism, s_ofs, fixed_s2, centers_s2, g2l_s2)

            pos_s2 = np.empty((len(remain_ids), 2), dtype=np.float64)
            pos_s2[:, 0] = canvas_w / 2.0
            pos_s2[:, 1] = canvas_h / 2.0
            pos_s2 = _solve(pos_s2, len(remain_ids), num_nets, net_starts, net_ends,
                            s_loc_s2, s_fix_s2, s_fpos_s2, s_ofs, net_size_thresh, b2b_pos=None)
            _clamp(pos_s2, remain_ids, half_w, half_h, canvas_w, canvas_h)

            for _ in range(b2b_iters):
                pos_s2 = _solve(pos_s2, len(remain_ids), num_nets, net_starts, net_ends,
                                s_loc_s2, s_fix_s2, s_fpos_s2, s_ofs, net_size_thresh, b2b_pos=pos_s2)
                _clamp(pos_s2, remain_ids, half_w, half_h, canvas_w, canvas_h)

            return _scatter_result(benchmark, num_macros, half_w, half_h, canvas_w, canvas_h,
                                   {tuple(anchor_ids): pos_s1, tuple(remain_ids): pos_s2}), np.empty(0, dtype=np.int64)

    # ── Standard single-stage path (anchor_fraction = 0) ────────────────────
    g2l = np.full(num_macros, -1, dtype=np.int64)
    g2l[movable_ids] = np.arange(n_mov)

    s_loc, s_fix, s_fpos = _classify_pins(
        s_mid, s_ism, s_ofs, fixed_mask_np, macro_centers, g2l)

    pos = np.empty((n_mov, 2), dtype=np.float64)
    pos[:, 0] = canvas_w / 2.0
    pos[:, 1] = canvas_h / 2.0
    pos = _solve(pos, n_mov, num_nets, net_starts, net_ends,
                 s_loc, s_fix, s_fpos, s_ofs, net_size_thresh, b2b_pos=None)
    _clamp(pos, movable_ids, half_w, half_h, canvas_w, canvas_h)

    for _ in range(b2b_iters):
        pos = _solve(pos, n_mov, num_nets, net_starts, net_ends,
                     s_loc, s_fix, s_fpos, s_ofs, net_size_thresh, b2b_pos=pos)
        _clamp(pos, movable_ids, half_w, half_h, canvas_w, canvas_h)

    return _scatter_result(benchmark, num_macros, half_w, half_h, canvas_w, canvas_h,
                           {tuple(movable_ids): pos}), np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_anchors(movable_ids, macro_ids_np, is_macro_np, net_ids_np,
                    half_w, half_h, anchor_fraction):
    """
    Return global IDs of the top anchor_fraction movable macros by area*net_degree.
    Always leaves at least one non-anchor movable.
    """
    num_macros = len(half_w)

    macro_mask = is_macro_np & (macro_ids_np >= 0)
    if macro_mask.any():
        pairs = np.unique(
            np.stack([macro_ids_np[macro_mask], net_ids_np[macro_mask]], axis=1), axis=0)
        net_degree = np.bincount(pairs[:, 0], minlength=num_macros).astype(np.float64)
    else:
        net_degree = np.zeros(num_macros, dtype=np.float64)

    area   = (half_w * half_h).astype(np.float64)
    scores = area * net_degree

    n_anchors = max(1, int(round(len(movable_ids) * anchor_fraction)))
    n_anchors = min(n_anchors, len(movable_ids) - 1)

    ranked = np.argsort(scores[movable_ids])[::-1]
    return movable_ids[ranked[:n_anchors]]


def _classify_pins(s_mid, s_ism, s_ofs, fixed_mask, macro_centers, g2l):
    """
    Classify sorted pins as fixed or movable given a fixed_mask and g2l mapping.
    Returns (s_loc, s_fix, s_fpos) ready for _solve.
    """
    pin_safe = np.where(~s_ism, 0, s_mid)
    s_fix    = ~s_ism | fixed_mask[pin_safe]
    s_loc    = np.where(s_fix, np.int64(-1), g2l[pin_safe])
    s_fpos   = np.where(
        (~s_ism)[:, None],
        s_ofs,
        macro_centers[pin_safe] + s_ofs,
    )
    return s_loc, s_fix, s_fpos


def _scatter_result(benchmark, num_macros, half_w, half_h, canvas_w, canvas_h, id_pos_map):
    """Write solved positions back into a full [num_macros, 2] float tensor."""
    result = benchmark.macro_positions[:num_macros].clone().float()
    for ids, pos in id_pos_map.items():
        ids = np.array(ids)
        if len(ids):
            result[ids] = torch.from_numpy(pos.astype(np.float32))
    hw = torch.from_numpy(half_w.astype(np.float32))
    hh = torch.from_numpy(half_h.astype(np.float32))
    result[:, 0] = result[:, 0].clamp(hw, canvas_w - hw)
    result[:, 1] = result[:, 1].clamp(hh, canvas_h - hh)
    return result


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
            if pi != min_p:
                _add_spring(loc[pi], loc[min_p], fix[pi], fix[min_p],
                            ofs[pi, dim], ofs[min_p, dim],
                            fpos[pi, dim], fpos[min_p, dim], w,
                            rows, cols, vals, b)
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
