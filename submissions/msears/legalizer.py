"""
legalizer.py — Post-placement legalization for CometPlacer.

Public API:
    spiralLegalize(pos, benchmark) -> torch.Tensor
        Removes hard-macro overlaps via greedy spiral push-out.
        Soft macros (indices >= num_hard_macros) are never touched.
        Fixed macros are treated as immovable obstacles.
"""

import torch


def _hasOverlap(cx_a, cy_a, w_a, h_a, cx_b, cy_b, w_b, h_b):
    """Return True if two axis-aligned rectangles strictly overlap."""
    return (abs(cx_a - cx_b) < (w_a + w_b) / 2.0 and
            abs(cy_a - cy_b) < (h_a + h_b) / 2.0)


def _fitsCanvas(cx, cy, w, h, canvas_w, canvas_h):
    """Return True if the macro fits entirely within the canvas."""
    return (cx - w / 2.0 >= 0.0 and cx + w / 2.0 <= canvas_w and
            cy - h / 2.0 >= 0.0 and cy + h / 2.0 <= canvas_h)


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
    Remove hard-macro overlaps by spiral push-out.

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
        print(f"    legalize pass {pass_num}: {num_skipped} already legal, "
              f"{num_moved} moved, {num_fallback} fallback")

    return pos
