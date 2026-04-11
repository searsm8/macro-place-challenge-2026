"""
legalizer.py — Post-placement legalization for MSPlacer.

Public API:
    spiral_legalize(pos, benchmark) -> torch.Tensor
        Removes hard-macro overlaps via spiral push-out.
        Soft macros (indices >= num_hard_macros) are never touched.
        Fixed macros are treated as immovable obstacles.
"""

import torch


def _overlaps(cx_a, cy_a, w_a, h_a, cx_b, cy_b, w_b, h_b) -> bool:
    """Return True if two axis-aligned rectangles strictly overlap (not just touch)."""
    return (abs(cx_a - cx_b) < (w_a + w_b) / 2.0 and
            abs(cy_a - cy_b) < (h_a + h_b) / 2.0)


def _in_canvas(cx, cy, w, h, cw, ch) -> bool:
    """Return True if the macro fits entirely within the canvas."""
    return (cx - w / 2.0 >= 0.0 and cx + w / 2.0 <= cw and
            cy - h / 2.0 >= 0.0 and cy + h / 2.0 <= ch)


def _spiral_candidates(cx0, cy0, r, step):
    """
    Yield (x, y) candidate positions on the perimeter of square ring r
    (half-side = r * step) centered on (cx0, cy0).

    Sweeps: bottom edge → right edge → top edge → left edge.
    """
    d = r * step
    eps = 1e-9  # float-rounding guard on loop bounds

    # Bottom edge: y fixed, x left → right
    y = cy0 - d
    x = cx0 - d
    while x <= cx0 + d + eps:
        yield x, y
        x += step

    # Right edge: x fixed, y bottom+step → top
    x = cx0 + d
    y = cy0 - d + step
    while y <= cy0 + d + eps:
        yield x, y
        y += step

    # Top edge: y fixed, x right-step → left
    y = cy0 + d
    x = cx0 + d - step
    while x >= cx0 - d - eps:
        yield x, y
        x -= step

    # Left edge: x fixed, y top-step → bottom+step
    x = cx0 - d
    y = cy0 + d - step
    while y >= cy0 - d + step - eps:
        yield x, y
        y -= step


def _count_overlaps(pos, n_hard, sizes):
    """Count overlapping hard macro pairs in the current placement."""
    count = 0
    for i in range(n_hard):
        cxi = float(pos[i, 0]); cyi = float(pos[i, 1])
        wi  = float(sizes[i, 0]); hi  = float(sizes[i, 1])
        for j in range(i + 1, n_hard):
            if _overlaps(cxi, cyi, wi, hi,
                         float(pos[j, 0]), float(pos[j, 1]),
                         float(sizes[j, 0]), float(sizes[j, 1])):
                count += 1
    return count


def _one_pass(pos, benchmark, step, max_ring):
    """
    Run one sweep of spiral push-out over all hard macros.
    Macros are processed largest-first. Returns (pos, moved, skipped, fallback).
    """
    n_hard = benchmark.num_hard_macros
    cw     = float(benchmark.canvas_width)
    ch     = float(benchmark.canvas_height)
    sizes  = benchmark.macro_sizes

    areas = sizes[:n_hard, 0] * sizes[:n_hard, 1]
    order = torch.argsort(areas, descending=True).tolist()

    placed_cx = []
    placed_cy = []
    placed_w  = []
    placed_h  = []

    moved = skipped = fallback = 0

    for idx in order:
        cx = float(pos[idx, 0])
        cy = float(pos[idx, 1])
        w  = float(sizes[idx, 0])
        h  = float(sizes[idx, 1])

        if benchmark.macro_fixed[idx]:
            placed_cx.append(cx); placed_cy.append(cy)
            placed_w.append(w);   placed_h.append(h)
            continue

        conflict = not _in_canvas(cx, cy, w, h, cw, ch)
        if not conflict:
            for pcx, pcy, pw, ph in zip(placed_cx, placed_cy, placed_w, placed_h):
                if _overlaps(cx, cy, w, h, pcx, pcy, pw, ph):
                    conflict = True
                    break

        if not conflict:
            skipped += 1
            placed_cx.append(cx); placed_cy.append(cy)
            placed_w.append(w);   placed_h.append(h)
            continue

        found = False
        for r in range(1, max_ring + 1):
            for ncx, ncy in _spiral_candidates(cx, cy, r, step):
                if not _in_canvas(ncx, ncy, w, h, cw, ch):
                    continue
                ok = True
                for pcx, pcy, pw, ph in zip(placed_cx, placed_cy, placed_w, placed_h):
                    if _overlaps(ncx, ncy, w, h, pcx, pcy, pw, ph):
                        ok = False
                        break
                if ok:
                    pos[idx, 0] = ncx
                    pos[idx, 1] = ncy
                    cx, cy = ncx, ncy
                    found = True
                    break
            if found:
                break

        if found:
            moved += 1
        else:
            cx = float(pos[idx, 0].clamp(w / 2.0, cw - w / 2.0))
            cy = float(pos[idx, 1].clamp(h / 2.0, ch - h / 2.0))
            pos[idx, 0] = cx
            pos[idx, 1] = cy
            fallback += 1

        placed_cx.append(cx); placed_cy.append(cy)
        placed_w.append(w);   placed_h.append(h)

    return pos, moved, skipped, fallback


def spiral_legalize(pos: torch.Tensor, benchmark, max_passes: int = 3) -> torch.Tensor:
    """
    Remove hard-macro overlaps by spiral push-out.

    Runs up to max_passes sweeps, stopping early if no overlaps remain.

    Args:
        pos        : [n, 2] float tensor of macro centers (hard + soft).
        benchmark  : Benchmark object with macro_sizes, macro_fixed, canvas dims.
        max_passes : Maximum number of sweep passes (default 3).

    Returns:
        Cloned pos with hard macro overlaps removed.
    """
    pos = pos.clone()

    n_hard = benchmark.num_hard_macros
    if n_hard == 0:
        return pos

    sizes    = benchmark.macro_sizes
    min_dim  = float(sizes[:n_hard].min())
    step     = min_dim * 0.5
    max_ring = 200

    for pass_num in range(1, max_passes + 1):
        remaining = _count_overlaps(pos, n_hard, sizes)
        if remaining == 0:
            break
        pos, moved, skipped, fallback = _one_pass(pos, benchmark, step, max_ring)
        print(f"    legalize pass {pass_num}: {skipped} already legal, "
              f"{moved} moved, {fallback} fallback")

    return pos
