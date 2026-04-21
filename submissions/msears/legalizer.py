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
# Geometry helpers
# ---------------------------------------------------------------------------

def _hasOverlap(cx_a, cy_a, w_a, h_a, cx_b, cy_b, w_b, h_b):
    """Return True if two axis-aligned rectangles strictly overlap."""
    return (abs(cx_a - cx_b) < (w_a + w_b) / 2.0 and
            abs(cy_a - cy_b) < (h_a + h_b) / 2.0)


def _fitsCanvas(cx, cy, w, h, canvas_w, canvas_h):
    """Return True if the macro fits entirely within the canvas."""
    return (cx - w / 2.0 >= 0.0 and cx + w / 2.0 <= canvas_w and
            cy - h / 2.0 >= 0.0 and cy + h / 2.0 <= canvas_h)


def _countOverlaps(pos, num_hard, sizes):
    """Count overlapping hard macro pairs."""
    count = 0
    for i in range(num_hard):
        cxi, cyi = float(pos[i, 0]), float(pos[i, 1])
        wi, hi = float(sizes[i, 0]), float(sizes[i, 1])
        for j in range(i + 1, num_hard):
            if _hasOverlap(cxi, cyi, wi, hi,
                           float(pos[j, 0]), float(pos[j, 1]),
                           float(sizes[j, 0]), float(sizes[j, 1])):
                count += 1
    return count


def _findOverlappingPairs(pos, num_hard, sizes):
    """
    Return a list of (severity, i, j) for every overlapping hard-macro pair.
    severity = min(ov_x, ov_y) — penetration depth along the easier-to-clear axis.
    """
    pairs = []
    for i in range(num_hard):
        cxi = float(pos[i, 0])
        cyi = float(pos[i, 1])
        wi  = float(sizes[i, 0])
        hi  = float(sizes[i, 1])
        for j in range(i + 1, num_hard):
            cxj = float(pos[j, 0])
            cyj = float(pos[j, 1])
            wj  = float(sizes[j, 0])
            hj  = float(sizes[j, 1])
            ov_x = (wi + wj) / 2.0 - abs(cxi - cxj)
            ov_y = (hi + hj) / 2.0 - abs(cyi - cyj)
            if ov_x > 0.0 and ov_y > 0.0:
                pairs.append((min(ov_x, ov_y), i, j))
    return pairs


# ---------------------------------------------------------------------------
# Bump legalization — Jacobi simultaneous pairwise SAT MTV
# ---------------------------------------------------------------------------

def _jacobiPass(pos, num_hard, phys_sizes, eff_sizes, fixed,
                canvas_w, canvas_h, margin, damping):
    """
    One Jacobi bump pass.

    Reads all positions once at the start of this pass, computes the
    minimum-translation-vector (MTV) push for every overlapping pair,
    accumulates net displacements per macro, then applies them all
    simultaneously with a damping factor.

    phys_sizes : physical macro dimensions — used for canvas clamping and
                 area-weighted split (actual mass matters for priority).
    eff_sizes  : effective dimensions including halo — used for overlap
                 detection so the legalizer enforces the same clearance as
                 the density force.

    Returns (num_overlapping_pairs, num_unresolvable).
    """
    delta_x = [0.0] * num_hard
    delta_y = [0.0] * num_hard
    count = 0
    unresolvable = 0

    for i in range(num_hard):
        cx_i = float(pos[i, 0])
        cy_i = float(pos[i, 1])
        wei = float(eff_sizes[i, 0])   # effective width  (halo-inflated)
        hei = float(eff_sizes[i, 1])   # effective height (halo-inflated)
        fi = bool(fixed[i])

        for j in range(i + 1, num_hard):
            cx_j = float(pos[j, 0])
            cy_j = float(pos[j, 1])
            wej = float(eff_sizes[j, 0])
            hej = float(eff_sizes[j, 1])

            ov_x = (wei + wej) / 2.0 - abs(cx_i - cx_j)
            ov_y = (hei + hej) / 2.0 - abs(cy_i - cy_j)

            if ov_x <= 0.0 or ov_y <= 0.0:
                continue

            count += 1
            fj = bool(fixed[j])

            if fi and fj:
                unresolvable += 1
                continue

            # Direction: push i away from j
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            if dx == 0.0 and dy == 0.0:
                dx = 1.0  # coincident: break tie toward +x

            sign_x = 1.0 if dx >= 0.0 else -1.0
            sign_y = 1.0 if dy >= 0.0 else -1.0

            # Minimum translation vector: push along the axis with smaller overlap
            if ov_x <= ov_y:
                push_x = sign_x * (ov_x + margin)
                push_y = 0.0
            else:
                push_x = 0.0
                push_y = sign_y * (ov_y + margin)

            # Area-weighted split using physical sizes (actual mass, not inflated)
            ai = float(phys_sizes[i, 0]) * float(phys_sizes[i, 1])
            aj = float(phys_sizes[j, 0]) * float(phys_sizes[j, 1])
            total_area = ai + aj

            if not fi and not fj:
                frac_i = aj / total_area
                frac_j = ai / total_area
            elif not fi:
                frac_i, frac_j = 1.0, 0.0
            else:
                frac_i, frac_j = 0.0, 1.0

            delta_x[i] += push_x * frac_i
            delta_y[i] += push_y * frac_i
            delta_x[j] -= push_x * frac_j
            delta_y[j] -= push_y * frac_j

    # Apply net displacements simultaneously — clamp to physical canvas bounds
    for i in range(num_hard):
        if bool(fixed[i]):
            continue
        if delta_x[i] == 0.0 and delta_y[i] == 0.0:
            continue
        wpi = float(phys_sizes[i, 0])
        hpi = float(phys_sizes[i, 1])
        new_x = float(pos[i, 0]) + damping * delta_x[i]
        new_y = float(pos[i, 1]) + damping * delta_y[i]
        pos[i, 0] = max(wpi / 2.0, min(canvas_w - wpi / 2.0, new_x))
        pos[i, 1] = max(hpi / 2.0, min(canvas_h - hpi / 2.0, new_y))

    return count, unresolvable


def _gaussSeidelCleanup(pos, num_hard, phys_sizes, eff_sizes, fixed,
                         canvas_w, canvas_h, margin, max_cleanup_passes=300):
    """
    Targeted Gauss-Seidel cleanup for a small number of remaining overlaps.

    Uses eff_sizes for overlap detection and phys_sizes for clamping/area,
    consistent with _jacobiPass.  Returns the number of physically overlapping
    pairs remaining (using phys_sizes, not eff_sizes, so the caller can decide
    whether a spiral fallback is needed for contest legality).
    """
    for _ in range(1, max_cleanup_passes + 1):
        pairs = _findOverlappingPairs(pos, num_hard, eff_sizes)
        if not pairs:
            break

        pairs.sort(reverse=True)   # deepest halo-overlap first

        for _, i, j in pairs:
            cx_i = float(pos[i, 0])
            cy_i = float(pos[i, 1])
            wei  = float(eff_sizes[i, 0])
            hei  = float(eff_sizes[i, 1])
            cx_j = float(pos[j, 0])
            cy_j = float(pos[j, 1])
            wej  = float(eff_sizes[j, 0])
            hej  = float(eff_sizes[j, 1])

            ov_x = (wei + wej) / 2.0 - abs(cx_i - cx_j)
            ov_y = (hei + hej) / 2.0 - abs(cy_i - cy_j)
            if ov_x <= 0.0 or ov_y <= 0.0:
                continue

            fi = bool(fixed[i])
            fj = bool(fixed[j])
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

            ai = float(phys_sizes[i, 0]) * float(phys_sizes[i, 1])
            aj = float(phys_sizes[j, 0]) * float(phys_sizes[j, 1])
            total_area = ai + aj

            if not fi and not fj:
                frac_i = aj / total_area
                frac_j = ai / total_area
            elif not fi:
                frac_i, frac_j = 1.0, 0.0
            else:
                frac_i, frac_j = 0.0, 1.0

            wpi = float(phys_sizes[i, 0])
            hpi = float(phys_sizes[i, 1])
            new_xi = max(wpi / 2.0, min(canvas_w - wpi / 2.0, cx_i + push_x * frac_i))
            new_yi = max(hpi / 2.0, min(canvas_h - hpi / 2.0, cy_i + push_y * frac_i))
            pos[i, 0] = new_xi
            pos[i, 1] = new_yi

            wpj = float(phys_sizes[j, 0])
            hpj = float(phys_sizes[j, 1])
            new_xj = max(wpj / 2.0, min(canvas_w - wpj / 2.0, cx_j - push_x * frac_j))
            new_yj = max(hpj / 2.0, min(canvas_h - hpj / 2.0, cy_j - push_y * frac_j))
            pos[j, 0] = new_xj
            pos[j, 1] = new_yj

    # Return physical overlap count — spiral fallback only needed for true collisions
    return _countOverlaps(pos, num_hard, phys_sizes)


def bumpLegalize(pos, benchmark, halo_size=0.0, max_passes=80, margin_frac=5e-3,
                 damping=0.5, stall_patience=10, margin_escalation=3.0,
                 fallback=True):
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

    phys_sizes = benchmark.macro_sizes          # physical dimensions
    eff_sizes  = phys_sizes * (1.0 + halo_size) # halo-inflated for overlap detection
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    min_dim = float(phys_sizes[:num_hard].min())
    margin_base = min_dim * margin_frac
    margin = margin_base
    fixed = benchmark.macro_fixed[:num_hard]

    best_count = float("inf")
    best_count_pass = 0

    for pass_num in range(1, max_passes + 1):
        count, unresolvable = _jacobiPass(
            pos, num_hard, phys_sizes, eff_sizes, fixed,
            canvas_w, canvas_h, margin, damping)

        if count == 0:
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

        print(f"    legalize bump pass {pass_num:3d}: "
              f"{count} overlapping pairs"
              f"{f'  ({unresolvable} both-fixed)' if unresolvable else ''}"
              f"{status}", flush=True)

    # Phase 2: Gauss-Seidel cleanup for the small residual.
    # Check with eff_sizes — if halo-pairs remain, clean them up.
    remaining_halo = len(_findOverlappingPairs(pos, num_hard, eff_sizes))
    if remaining_halo > 0:
        cleanup_margin = min_dim * 0.02
        print(f"    legalize bump: {remaining_halo} halo-pairs remain — "
              f"Gauss-Seidel cleanup (margin={cleanup_margin:.5f})", flush=True)
        remaining_phys = _gaussSeidelCleanup(pos, num_hard, phys_sizes, eff_sizes,
                                              fixed, canvas_w, canvas_h, cleanup_margin)
    else:
        remaining_phys = _countOverlaps(pos, num_hard, phys_sizes)

    if remaining_phys > 0:
        print(f"    legalize bump: {remaining_phys} physical overlaps remain after cleanup", flush=True)
        if fallback:
            print(f"    legalize bump: falling back to spiral", flush=True)
            pos = spiralLegalize(pos, benchmark, max_passes=3)

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

    placed_cx, placed_cy = [], []
    placed_w, placed_h = [], []
    num_moved = num_skipped = num_fallback = 0

    for idx in order:
        cx = float(pos[idx, 0])
        cy = float(pos[idx, 1])
        w = float(sizes[idx, 0])
        h = float(sizes[idx, 1])

        # Fixed macros are immovable obstacles
        if benchmark.macro_fixed[idx]:
            placed_cx.append(cx)
            placed_cy.append(cy)
            placed_w.append(w)
            placed_h.append(h)
            continue

        # Check if current position is conflict-free
        conflict = not _fitsCanvas(cx, cy, w, h, canvas_w, canvas_h)
        if not conflict:
            for pcx, pcy, pw, ph in zip(placed_cx, placed_cy, placed_w, placed_h):
                if _hasOverlap(cx, cy, w, h, pcx, pcy, pw, ph):
                    conflict = True
                    break

        if not conflict:
            num_skipped += 1
            placed_cx.append(cx)
            placed_cy.append(cy)
            placed_w.append(w)
            placed_h.append(h)
            continue

        # Spiral search for conflict-free position
        found = _searchSpiral(pos, idx, cx, cy, w, h, canvas_w, canvas_h,
                              step, max_ring, placed_cx, placed_cy,
                              placed_w, placed_h)

        if found:
            cx = float(pos[idx, 0])
            cy = float(pos[idx, 1])
            num_moved += 1
        else:
            cx = float(pos[idx, 0].clamp(w / 2.0, canvas_w - w / 2.0))
            cy = float(pos[idx, 1].clamp(h / 2.0, canvas_h - h / 2.0))
            pos[idx, 0] = cx
            pos[idx, 1] = cy
            num_fallback += 1

        placed_cx.append(cx)
        placed_cy.append(cy)
        placed_w.append(w)
        placed_h.append(h)

    return pos, num_moved, num_skipped, num_fallback


def _searchSpiral(pos, idx, cx, cy, w, h, canvas_w, canvas_h,
                  step, max_ring, placed_cx, placed_cy, placed_w, placed_h):
    """Search outward in spiral rings for a conflict-free position."""
    for ring in range(1, max_ring + 1):
        for new_cx, new_cy in _spiralCandidates(cx, cy, ring, step):
            if not _fitsCanvas(new_cx, new_cy, w, h, canvas_w, canvas_h):
                continue
            ok = True
            for pcx, pcy, pw, ph in zip(placed_cx, placed_cy, placed_w, placed_h):
                if _hasOverlap(new_cx, new_cy, w, h, pcx, pcy, pw, ph):
                    ok = False
                    break
            if ok:
                pos[idx, 0] = new_cx
                pos[idx, 1] = new_cy
                return True
    return False


def spiralLegalize(pos, benchmark, max_passes=3):
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
        print(f"    legalize spiral pass {pass_num}: {num_skipped} already legal, "
              f"{num_moved} moved, {num_fallback} fallback", flush=True)

    return pos
