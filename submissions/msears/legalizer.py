"""
legalizer.py — Post-placement legalization for CometPlacer.

Public API:
    bumpLegalize(pos, benchmark) -> torch.Tensor
        Removes hard-macro overlaps via iterative Jacobi-style pairwise
        minimum-displacement bumping (SAT MTV). Soft macros are never touched.
        Fixed macros are treated as immovable obstacles.

    spiralLegalize(pos, benchmark) -> torch.Tensor
        Legacy greedy spiral push-out. Kept as a last-resort fallback.
"""

import torch


# ---------------------------------------------------------------------------
# Vectorized geometry helpers
# ---------------------------------------------------------------------------

def _countOverlaps(pos, num_hard, sizes, upper=None):
    """Count overlapping hard macro pairs."""
    cx = pos[:num_hard, 0]
    cy = pos[:num_hard, 1]
    w  = sizes[:num_hard, 0]
    h  = sizes[:num_hard, 1]
    dx = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs()
    dy = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs()
    ov = ((w.unsqueeze(1) + w.unsqueeze(0)) * 0.5 - dx > 0) & \
         ((h.unsqueeze(1) + h.unsqueeze(0)) * 0.5 - dy > 0)
    if upper is None:
        upper = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
    return int((ov & upper).sum())


def _findOverlappingPairs(pos, num_hard, sizes, upper=None):
    """
    Return a list of (severity, i, j) for every overlapping pair.
    severity = min(ov_x, ov_y) — penetration depth along the easier-to-clear axis.
    """
    cx = pos[:num_hard, 0]
    cy = pos[:num_hard, 1]
    w  = sizes[:num_hard, 0]
    h  = sizes[:num_hard, 1]
    dx = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs()
    dy = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs()
    ov_x = (w.unsqueeze(1) + w.unsqueeze(0)) * 0.5 - dx
    ov_y = (h.unsqueeze(1) + h.unsqueeze(0)) * 0.5 - dy
    if upper is None:
        upper = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
    mask = (ov_x > 0) & (ov_y > 0) & upper
    i_idx, j_idx = mask.nonzero(as_tuple=True)
    if i_idx.numel() == 0:
        return []
    sev = torch.minimum(ov_x[i_idx, j_idx], ov_y[i_idx, j_idx])
    return list(zip(sev.tolist(), i_idx.tolist(), j_idx.tolist()))


def _fitsCanvas(cx, cy, w, h, canvas_w, canvas_h):
    """Return True if the macro fits entirely within the canvas."""
    return (cx - w / 2.0 >= 0.0 and cx + w / 2.0 <= canvas_w and
            cy - h / 2.0 >= 0.0 and cy + h / 2.0 <= canvas_h)


# ---------------------------------------------------------------------------
# Bump legalization — Jacobi simultaneous pairwise SAT MTV
# ---------------------------------------------------------------------------

def _jacobiPass(pos, n, half_sum_ew, half_sum_eh, upper,
                pw, ph, pw_half, ph_half, not_fixed, fixed,
                canvas_w, canvas_h, margin, damping):
    """
    One vectorized Jacobi bump pass.

    Precomputed pass-invariant args (built once in bumpLegalize):
      half_sum_ew/eh : (ew[i]+ew[j])/2 outer sums  [n, n]
      upper          : upper-triangle bool mask      [n, n]
      pw/ph          : physical sizes                [n]
      pw_half/ph_half: physical half-sizes           [n]
      not_fixed      : ~fixed bool tensor            [n]
      fixed          : fixed flags bool tensor       [n]

    Returns (num_overlapping_pairs, num_unresolvable).
    """
    cx = pos[:n, 0]
    cy = pos[:n, 1]
    dx_mat = cx.unsqueeze(1) - cx.unsqueeze(0)
    dy_mat = cy.unsqueeze(1) - cy.unsqueeze(0)
    ov_x_mat = half_sum_ew - dx_mat.abs()
    ov_y_mat = half_sum_eh - dy_mat.abs()
    mask = (ov_x_mat > 0) & (ov_y_mat > 0) & upper
    i_idx, j_idx = mask.nonzero(as_tuple=True)
    count = i_idx.numel()
    if count == 0:
        return 0, 0

    ov_x_p = ov_x_mat[i_idx, j_idx]
    ov_y_p = ov_y_mat[i_idx, j_idx]
    dxp    = dx_mat[i_idx, j_idx]
    dyp    = dy_mat[i_idx, j_idx]

    coincident = (dxp == 0) & (dyp == 0)
    dxp = torch.where(coincident, torch.ones_like(dxp), dxp)
    sign_x = dxp.sign()
    sign_y = dyp.sign()

    push_x_axis = ov_x_p <= ov_y_p
    push_x = torch.where(push_x_axis, sign_x * (ov_x_p + margin), torch.zeros_like(ov_x_p))
    push_y = torch.where(~push_x_axis, sign_y * (ov_y_p + margin), torch.zeros_like(ov_y_p))

    ai = pw[i_idx] * ph[i_idx]
    aj = pw[j_idx] * ph[j_idx]
    total_a = ai + aj
    fi = fixed[i_idx]
    fj = fixed[j_idx]
    both_fixed = fi & fj
    frac_i = torch.where(fi, torch.zeros_like(ai),
              torch.where(fj, torch.ones_like(ai),  aj / total_a))
    frac_j = torch.where(fj, torch.zeros_like(aj),
              torch.where(fi, torch.ones_like(aj),  ai / total_a))

    delta_x = torch.zeros(n, dtype=pos.dtype)
    delta_y = torch.zeros(n, dtype=pos.dtype)
    delta_x.scatter_add_(0, i_idx, push_x * frac_i)
    delta_y.scatter_add_(0, i_idx, push_y * frac_i)
    delta_x.scatter_add_(0, j_idx, -push_x * frac_j)
    delta_y.scatter_add_(0, j_idx, -push_y * frac_j)

    new_x = (pos[:n, 0] + damping * delta_x).clamp(pw_half, canvas_w - pw_half)
    new_y = (pos[:n, 1] + damping * delta_y).clamp(ph_half, canvas_h - ph_half)
    pos[:n, 0] = torch.where(not_fixed, new_x, pos[:n, 0])
    pos[:n, 1] = torch.where(not_fixed, new_y, pos[:n, 1])

    return count, int(both_fixed.sum())


def _gaussSeidelCleanup(pos, num_hard, phys_sizes, eff_sizes, fixed,
                         canvas_w, canvas_h, margin, max_cleanup_passes=300):
    """
    Targeted Gauss-Seidel cleanup for a small number of remaining overlaps.

    Uses eff_sizes for overlap detection and phys_sizes for clamping/area,
    consistent with _jacobiPass.  Returns the number of physically overlapping
    pairs remaining (using phys_sizes, not eff_sizes).
    """
    pw = phys_sizes[:num_hard, 0]
    ph = phys_sizes[:num_hard, 1]
    ew = eff_sizes[:num_hard, 0]
    eh = eff_sizes[:num_hard, 1]

    # Precompute upper mask once (never changes).
    upper = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
    pairs = _findOverlappingPairs(pos, num_hard, eff_sizes, upper=upper)

    # pos[:num_hard].numpy() is a zero-copy view — mutations to pos_np propagate to pos.
    pos_np   = pos[:num_hard].numpy()
    ew_np    = ew.numpy()
    eh_np    = eh.numpy()
    pw_np    = pw.numpy()
    ph_np    = ph.numpy()
    fixed_np = fixed.bool().numpy()
    pw_half  = pw_np * 0.5
    ph_half  = ph_np * 0.5

    # Full O(n²) rebuild period — catches any new overlaps created by bumping.
    _REBUILD_EVERY = 20

    for pass_num in range(max_cleanup_passes):
        if not pairs:
            break
        pairs.sort(reverse=True)   # deepest halo-overlap first

        any_moved = False
        for _, i, j in pairs:
            cx_i = pos_np[i, 0]
            cy_i = pos_np[i, 1]
            cx_j = pos_np[j, 0]
            cy_j = pos_np[j, 1]

            ov_x = float(ew_np[i] + ew_np[j]) * 0.5 - abs(cx_i - cx_j)
            ov_y = float(eh_np[i] + eh_np[j]) * 0.5 - abs(cy_i - cy_j)
            if ov_x <= 0.0 or ov_y <= 0.0:
                continue

            fi = bool(fixed_np[i])
            fj = bool(fixed_np[j])
            if fi and fj:
                continue

            dx = cx_i - cx_j
            dy = cy_i - cy_j
            if dx == 0.0 and dy == 0.0:
                dx = 1.0

            sign_x = 1.0 if dx >= 0.0 else -1.0
            sign_y = 1.0 if dy >= 0.0 else -1.0

            if ov_x <= ov_y:
                push_x = sign_x * (ov_x + margin)
                push_y = 0.0
            else:
                push_x = 0.0
                push_y = sign_y * (ov_y + margin)

            ai = float(pw_np[i]) * float(ph_np[i])
            aj = float(pw_np[j]) * float(ph_np[j])
            total_a = ai + aj

            if not fi and not fj:
                frac_i = aj / total_a
                frac_j = ai / total_a
            elif not fi:
                frac_i, frac_j = 1.0, 0.0
            else:
                frac_i, frac_j = 0.0, 1.0

            pos_np[i, 0] = max(pw_half[i], min(canvas_w - pw_half[i], cx_i + push_x * frac_i))
            pos_np[i, 1] = max(ph_half[i], min(canvas_h - ph_half[i], cy_i + push_y * frac_i))
            pos_np[j, 0] = max(pw_half[j], min(canvas_w - pw_half[j], cx_j - push_x * frac_j))
            pos_np[j, 1] = max(ph_half[j], min(canvas_h - ph_half[j], cy_j - push_y * frac_j))
            any_moved = True

        if not any_moved:
            break   # all remaining pairs are both-fixed or geometry-degenerate

        # Rebuild pair list: full O(n²) scan every _REBUILD_EVERY passes to catch new
        # overlaps; otherwise a fast O(P) vectorized re-filter of the existing pair set.
        if (pass_num + 1) % _REBUILD_EVERY == 0:
            pairs = _findOverlappingPairs(pos, num_hard, eff_sizes, upper=upper)
        elif pairs:
            i_t = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            j_t = torch.tensor([p[2] for p in pairs], dtype=torch.long)
            ov_x_t = (ew[i_t] + ew[j_t]) * 0.5 - (pos[i_t, 0] - pos[j_t, 0]).abs()
            ov_y_t = (eh[i_t] + eh[j_t]) * 0.5 - (pos[i_t, 1] - pos[j_t, 1]).abs()
            keep   = ((ov_x_t > 0) & (ov_y_t > 0)).nonzero(as_tuple=True)[0]
            sev_t  = torch.minimum(ov_x_t, ov_y_t)
            pairs  = [(float(sev_t[k]), int(i_t[k]), int(j_t[k])) for k in keep.tolist()]

    return _countOverlaps(pos, num_hard, phys_sizes, upper=upper)


def bumpLegalize(pos, benchmark, halo_size=0.0, max_passes=80, margin_frac=5e-3,
                 damping=0.5, stall_patience=10, margin_escalation=3.0,
                 fallback=True, verbose=False, quiet=False):
    """
    Remove hard-macro overlaps via iterative Jacobi-style minimum-displacement bumping.

    Phase 1 — Jacobi main loop:
      Each pass scans all hard-macro pairs from start-of-pass positions, accumulates
      the SAT MTV push per macro (simultaneous Jacobi), then applies with damping.
      Stall is detected via best_count tracking; on stall the push margin is
      escalated so macros move further to escape local force-cancellation fixed-points.

    Phase 2 — Gauss-Seidel cleanup:
      After Jacobi reduces overlaps to a small residual, sequential (Gauss-Seidel)
      resolution breaks the remaining force-cancellation in dense clusters.  Each
      resolution immediately updates positions so later pairs benefit from prior bumps
      within the same pass.  Only the macros in the remaining pairs are moved.

    Phase 3 — Spiral fallback (optional):
      If both phases fail to achieve legality, falls back to spiralLegalize.

    Args:
        pos               : [n, 2] float tensor of macro centers
        benchmark         : Benchmark object
        halo_size         : fractional halo around each macro (same as density halo).
                            eff_width = phys_width * (1 + halo_size).  Overlap is
                            detected on the inflated footprint so the legalizer
                            enforces the same clearance the density force targets.
        max_passes        : Jacobi phase maximum passes (default 80)
        margin_frac       : initial post-resolution clearance as fraction of min dim
        damping           : fraction of computed displacement applied per pass (0 < d ≤ 1)
        stall_patience    : escalate margin after this many passes without best_count drop
        margin_escalation : multiply margin by this factor on stall (default 3×)
        fallback          : if True, run spiralLegalize if both phases leave physical overlaps

    Returns: cloned pos with hard macro overlaps removed.
    """
    pos = pos.clone()
    num_hard = benchmark.num_hard_macros
    if num_hard == 0:
        return pos

    n = num_hard
    phys_sizes = benchmark.macro_sizes          # physical dimensions
    eff_sizes  = phys_sizes * (1.0 + halo_size) # halo-inflated for overlap detection
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    min_dim = float(phys_sizes[:n].min())
    margin_base = min_dim * margin_frac
    margin = margin_base
    fixed = benchmark.macro_fixed[:n]

    # Precompute pass-invariant quantities once
    ew = eff_sizes[:n, 0]
    eh = eff_sizes[:n, 1]
    pw = phys_sizes[:n, 0]
    ph = phys_sizes[:n, 1]
    half_sum_ew = (ew.unsqueeze(1) + ew.unsqueeze(0)) * 0.5   # [n, n]
    half_sum_eh = (eh.unsqueeze(1) + eh.unsqueeze(0)) * 0.5
    upper       = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pw_half     = pw * 0.5
    ph_half     = ph * 0.5
    fixed_bool  = fixed.bool()
    not_fixed   = ~fixed_bool

    best_count = float("inf")
    best_count_pass = 0

    for pass_num in range(1, max_passes + 1):
        count, unresolvable = _jacobiPass(
            pos, n, half_sum_ew, half_sum_eh, upper,
            pw, ph, pw_half, ph_half, not_fixed, fixed_bool,
            canvas_w, canvas_h, margin, damping)

        if count == 0:
            if not quiet:
                print(f"    legalize bump: clean after {pass_num} Jacobi pass(es)", flush=True)
            return pos

        status = ""
        if count < best_count:
            best_count = count
            best_count_pass = pass_num
        elif pass_num - best_count_pass >= stall_patience:
            margin = min(margin * margin_escalation, min_dim * 0.5)
            best_count_pass = pass_num   # reset patience after escalation
            status = f"  [stalled → margin={margin:.5f}]"

        if verbose:
            print(f"    legalize bump pass {pass_num:3d}: "
                  f"{count} overlapping pairs"
                  f"{f'  ({unresolvable} both-fixed)' if unresolvable else ''}"
                  f"{status}", flush=True)

    # Phase 2: Gauss-Seidel cleanup for the small residual.
    remaining_halo = len(_findOverlappingPairs(pos, num_hard, eff_sizes, upper=upper))
    if remaining_halo > 0:
        cleanup_margin = min_dim * 0.02
        if not quiet:
            print(f"    legalize bump: {remaining_halo} halo-pairs remain — "
                  f"Gauss-Seidel cleanup (margin={cleanup_margin:.5f})", flush=True)
        remaining_phys = _gaussSeidelCleanup(pos, num_hard, phys_sizes, eff_sizes,
                                              fixed, canvas_w, canvas_h, cleanup_margin)
    else:
        remaining_phys = _countOverlaps(pos, num_hard, phys_sizes, upper=upper)

    if remaining_phys > 0:
        if not quiet:
            print(f"    legalize bump: {remaining_phys} physical overlaps remain after cleanup", flush=True)
        if fallback:
            if not quiet:
                print(f"    legalize bump: falling back to spiral", flush=True)
            pos = spiralLegalize(pos, benchmark, max_passes=3, verbose=verbose, quiet=quiet)

    return pos


# ---------------------------------------------------------------------------
# Spiral legalization (greedy push-out — kept as last-resort fallback)
# ---------------------------------------------------------------------------

def _spiralCandidates(center_x, center_y, ring, step):
    """
    Yield (x, y) candidate positions on square ring `ring` centered on
    (center_x, center_y). Sweeps: bottom -> right -> top -> left edges.
    """
    half_side = ring * step
    eps = 1e-9

    # Bottom edge
    y = center_y - half_side
    x = center_x - half_side
    while x <= center_x + half_side + eps:
        yield x, y
        x += step

    # Right edge
    x = center_x + half_side
    y = center_y - half_side + step
    while y <= center_y + half_side + eps:
        yield x, y
        y += step

    # Top edge
    y = center_y + half_side
    x = center_x + half_side - step
    while x >= center_x - half_side - eps:
        yield x, y
        x -= step

    # Left edge
    x = center_x - half_side
    y = center_y + half_side - step
    while y >= center_y - half_side + step - eps:
        yield x, y
        y -= step


def _runOnePass(pos, benchmark, step, max_ring):
    """
    Run one sweep of spiral push-out over all hard macros.
    Macros are processed largest-first (descending area).
    Returns (pos, num_moved, num_skipped, num_fallback).
    """
    num_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes

    areas = sizes[:num_hard, 0] * sizes[:num_hard, 1]
    order = torch.argsort(areas, descending=True).tolist()

    # Preallocate placed-macro arrays for vectorized conflict check
    placed_cx = torch.zeros(num_hard, dtype=sizes.dtype)
    placed_cy = torch.zeros(num_hard, dtype=sizes.dtype)
    placed_w  = torch.zeros(num_hard, dtype=sizes.dtype)
    placed_h  = torch.zeros(num_hard, dtype=sizes.dtype)
    num_placed = 0

    num_moved = num_skipped = num_fallback = 0

    for idx in order:
        cx = float(pos[idx, 0])
        cy = float(pos[idx, 1])
        w  = float(sizes[idx, 0])
        h  = float(sizes[idx, 1])

        # Fixed macros are immovable obstacles
        if benchmark.macro_fixed[idx]:
            placed_cx[num_placed] = cx
            placed_cy[num_placed] = cy
            placed_w[num_placed]  = w
            placed_h[num_placed]  = h
            num_placed += 1
            continue

        def _conflict(tx, ty):
            # Canvas bounds check
            if (tx - w * 0.5 < 0.0 or tx + w * 0.5 > canvas_w or
                    ty - h * 0.5 < 0.0 or ty + h * 0.5 > canvas_h):
                return True
            if num_placed == 0:
                return False
            # Vectorized overlap check against all placed macros
            ov_x = (placed_cx[:num_placed] - tx).abs()
            ov_y = (placed_cy[:num_placed] - ty).abs()
            return bool(((((placed_w[:num_placed] + w) * 0.5 - ov_x) > 0) &
                          (((placed_h[:num_placed] + h) * 0.5 - ov_y) > 0)).any())

        if not _conflict(cx, cy):
            num_skipped += 1
        else:
            found = False
            for ring in range(1, max_ring + 1):
                for new_cx, new_cy in _spiralCandidates(cx, cy, ring, step):
                    if not _conflict(new_cx, new_cy):
                        pos[idx, 0] = new_cx
                        pos[idx, 1] = new_cy
                        cx, cy = new_cx, new_cy
                        found = True
                        break
                if found:
                    break

            if found:
                num_moved += 1
            else:
                cx = float(max(w * 0.5, min(canvas_w - w * 0.5, float(pos[idx, 0]))))
                cy = float(max(h * 0.5, min(canvas_h - h * 0.5, float(pos[idx, 1]))))
                pos[idx, 0] = cx
                pos[idx, 1] = cy
                num_fallback += 1

        placed_cx[num_placed] = cx
        placed_cy[num_placed] = cy
        placed_w[num_placed]  = w
        placed_h[num_placed]  = h
        num_placed += 1

    return pos, num_moved, num_skipped, num_fallback


def spiralLegalize(pos, benchmark, max_passes=3, verbose=False, quiet=False):
    """
    Remove hard-macro overlaps by greedy spiral push-out (legacy).

    Runs up to max_passes sweeps. Exits early if zero overlaps remain.
    Soft macros are never moved.

    Args:
        pos        : [n, 2] float tensor of macro centers
        benchmark  : Benchmark object
        max_passes : maximum sweep passes (default 3)

    Returns: cloned pos with hard macro overlaps removed.
    """
    pos = pos.clone()
    num_hard = benchmark.num_hard_macros
    if num_hard == 0:
        return pos

    sizes = benchmark.macro_sizes
    min_dim = float(sizes[:num_hard].min())
    step = min_dim * 0.5
    max_ring = 200

    for pass_num in range(1, max_passes + 1):
        remaining = _countOverlaps(pos, num_hard, sizes)
        if remaining == 0:
            break
        pos, num_moved, num_skipped, num_fallback = _runOnePass(
            pos, benchmark, step, max_ring)
        if verbose:
            print(f"    legalize spiral pass {pass_num}: {num_skipped} already legal, "
                  f"{num_moved} moved, {num_fallback} fallback", flush=True)

    return pos
