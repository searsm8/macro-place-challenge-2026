"""
proxy.py — Independent proxy cost evaluator.

Replicates the three terms of the contest proxy without using PlacementCost:

  proxy = 1.0 × wl_cost + 0.5 × density_cost + 0.5 × congestion_cost

All three functions accept (pos [N,2], net_data, benchmark) on CPU.

Accuracy vs. harness
--------------------
  wl_cost       exact match  — same weighted HPWL formula and normalisation
  density_cost  exact match  — same grid, same exact-overlap kernel, same abu(10%)
  congestion_cost  approximate — RUDY H/V instead of harness L/T-shaped routing;
                    will diverge on congested benchmarks where macro blockage matters

Use validate() to measure the current deviation from the harness on a live plc.
"""

import math
import torch
from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pin_positions(pos: torch.Tensor, net_data: dict) -> torch.Tensor:
    """Compute absolute pin positions [P, 2] from macro positions."""
    macro_ids = net_data["macro_ids"]   # [P]
    offsets   = net_data["offsets"]     # [P, 2]
    is_macro  = net_data["is_macro"]    # [P] bool
    safe_ids  = macro_ids.clamp(min=0)
    mflag     = is_macro.float().unsqueeze(1)
    return pos[safe_ids] * mflag + offsets  # [P, 2]


def _per_net_bbox(pin_pos: torch.Tensor, net_data: dict):
    """Return (x_min, x_max, y_min, y_max) each [K] from pin positions."""
    K       = net_data["num_nets"]
    net_ids = net_data["net_ids"]
    big     = pin_pos.new_full((K,), float("inf"))
    neg     = pin_pos.new_full((K,), float("-inf"))
    x_min   = big.clone().scatter_reduce_(0, net_ids, pin_pos[:, 0], reduce="amin", include_self=True)
    x_max   = neg.clone().scatter_reduce_(0, net_ids, pin_pos[:, 0], reduce="amax", include_self=True)
    y_min   = big.clone().scatter_reduce_(0, net_ids, pin_pos[:, 1], reduce="amin", include_self=True)
    y_max   = neg.clone().scatter_reduce_(0, net_ids, pin_pos[:, 1], reduce="amax", include_self=True)
    return x_min, x_max, y_min, y_max


def _abu(values: torch.Tensor, frac: float) -> float:
    """Average of top-frac of values (same as harness abu())."""
    n    = values.numel()
    k    = max(1, math.floor(n * frac))
    topk = torch.topk(values.float(), k, largest=True).values
    return topk.mean().item()


# ---------------------------------------------------------------------------
# WL cost
# ---------------------------------------------------------------------------

def wl_cost(pos: torch.Tensor, net_data: dict, benchmark: Benchmark) -> float:
    """
    Net-weighted exact HPWL normalised by canvas perimeter × total net count.

    Matches harness get_cost():
      wl_cost = Σ_k(weight_k × HPWL_k) / ((W + H) × net_cnt)

    net_cnt includes skipped (< 2 pin) nets, matching the harness denominator.
    """
    with torch.no_grad():
        pin_pos               = _pin_positions(pos, net_data)
        x_min, x_max, y_min, y_max = _per_net_bbox(pin_pos, net_data)
        hpwl_k                = (x_max - x_min) + (y_max - y_min)   # [K]
        weights               = net_data.get("net_weights",
                                             torch.ones(net_data["num_nets"]))
        weighted_hpwl         = (weights * hpwl_k).sum().item()

    W = float(benchmark.canvas_width)
    H = float(benchmark.canvas_height)
    # plc_net_cnt = Σ weight_k over all nets with sinks (matches harness net_cnt).
    net_cnt = net_data.get("plc_net_cnt", net_data["num_nets"] + net_data["num_skipped"])
    return weighted_hpwl / ((W + H) * net_cnt)


# ---------------------------------------------------------------------------
# Density cost
# ---------------------------------------------------------------------------

def density_cost(pos: torch.Tensor, benchmark: Benchmark) -> float:
    """
    Macro area density on the benchmark grid, abu(top 10%) × 0.5.

    Matches harness get_density_cost() / get_grid_cells_density():
      - exact rectangular overlap per cell (no clamping, no area-ratio scaling)
      - grid = benchmark.grid_rows × benchmark.grid_cols
      - cell density = overlap_area / cell_area
      - cost = 0.5 × mean(top 10% non-zero cells)
    """
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    W    = float(benchmark.canvas_width)
    H    = float(benchmark.canvas_height)
    cw   = W / cols
    ch   = H / rows

    sizes = benchmark.macro_sizes.to(pos)    # [N, 2]
    x_lo  = pos[:, 0] - sizes[:, 0] / 2
    x_hi  = pos[:, 0] + sizes[:, 0] / 2
    y_lo  = pos[:, 1] - sizes[:, 1] / 2
    y_hi  = pos[:, 1] + sizes[:, 1] / 2

    col_edges = torch.linspace(0.0, W, cols + 1, dtype=pos.dtype)  # [C+1]
    row_edges = torch.linspace(0.0, H, rows + 1, dtype=pos.dtype)  # [R+1]

    # Fractional overlap [N, C] and [N, R]
    ox = (torch.minimum(x_hi.unsqueeze(1), col_edges[1:].unsqueeze(0))
          - torch.maximum(x_lo.unsqueeze(1), col_edges[:-1].unsqueeze(0))).clamp(min=0.0)
    oy = (torch.minimum(y_hi.unsqueeze(1), row_edges[1:].unsqueeze(0))
          - torch.maximum(y_lo.unsqueeze(1), row_edges[:-1].unsqueeze(0))).clamp(min=0.0)

    # density[r,c] = Σ_m oy[m,r]*ox[m,c] / (cw*ch)
    density_map = (oy.T @ ox) / (cw * ch)   # [R, C]

    return 0.5 * _abu(density_map.flatten(), 0.10)


# ---------------------------------------------------------------------------
# Congestion cost
# ---------------------------------------------------------------------------

def congestion_cost(pos: torch.Tensor, net_data: dict, benchmark: Benchmark) -> float:
    """
    RUDY H/V routing demand, capacity-normalised, abu(top 5%).

    Approximates harness get_congestion_cost():
      - H demand per cell ∝ bbox_h / bbox_area  (horizontal wires cross vertical edges)
      - V demand per cell ∝ bbox_w / bbox_area  (vertical wires cross horizontal edges)
      - Capacity: h_cap = hroutes_per_micron × cell_h,  v_cap = vroutes_per_micron × cell_w
      - cost = abu(top 5% of  V_map/v_cap + H_map/h_cap)

    Deviation from harness: harness routes L/T-shaped paths and adds macro routing
    blockage; RUDY distributes demand uniformly over the bbox. Expect ~10-20%
    difference on heavily congested benchmarks.
    """
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    W    = float(benchmark.canvas_width)
    H    = float(benchmark.canvas_height)
    cw   = W / cols
    ch   = H / rows

    h_cap = benchmark.hroutes_per_micron * ch   # H routing tracks per cell
    v_cap = benchmark.vroutes_per_micron  * cw   # V routing tracks per cell

    with torch.no_grad():
        pin_pos               = _pin_positions(pos, net_data)
        x_min, x_max, y_min, y_max = _per_net_bbox(pin_pos, net_data)

    bbox_w  = (x_max - x_min).clamp(min=1e-6)
    bbox_h  = (y_max - y_min).clamp(min=1e-6)
    weights = net_data.get("net_weights", torch.ones(net_data["num_nets"]))

    # H demand density = weight × bbox_h / bbox_area  (wires per unit area)
    h_demand = weights * bbox_h / (bbox_w * bbox_h)   # = weight / bbox_w
    # V demand density = weight × bbox_w / bbox_area  = weight / bbox_h
    v_demand = weights * bbox_w / (bbox_w * bbox_h)

    col_edges = torch.linspace(0.0, W, cols + 1)
    row_edges = torch.linspace(0.0, H, rows + 1)

    # Fractional bbox overlap per cell [K, C] and [K, R]
    ox = (torch.minimum(x_max.unsqueeze(1), col_edges[1:].unsqueeze(0))
          - torch.maximum(x_min.unsqueeze(1), col_edges[:-1].unsqueeze(0))).clamp(min=0.0)
    oy = (torch.minimum(y_max.unsqueeze(1), row_edges[1:].unsqueeze(0))
          - torch.maximum(y_min.unsqueeze(1), row_edges[:-1].unsqueeze(0))).clamp(min=0.0)

    # H map [R,C]: horizontal wire demand spread uniformly over bbox rows × cols
    h_map = (oy.T @ (h_demand.unsqueeze(1) * ox)) / (cw * ch)
    # V map [R,C]: vertical wire demand
    v_map = (oy.T @ (v_demand.unsqueeze(1) * ox)) / (cw * ch)

    combined = v_map / v_cap + h_map / h_cap   # [R, C]
    return _abu(combined.flatten(), 0.05)


# ---------------------------------------------------------------------------
# Combined proxy
# ---------------------------------------------------------------------------

def compute_proxy(
    pos: torch.Tensor,
    net_data: dict,
    benchmark: Benchmark,
) -> dict:
    """
    Independent proxy cost.  Returns the same keys as harness compute_proxy_cost().

    proxy = 1.0 × wl + 0.5 × density + 0.5 × congestion
    """
    pos_cpu      = pos.detach().cpu()
    nd_cpu       = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in net_data.items()}
    bm_cpu_sizes = benchmark.macro_sizes.cpu()

    wl   = wl_cost(pos_cpu, nd_cpu, benchmark)
    den  = density_cost(pos_cpu, benchmark)
    cong = congestion_cost(pos_cpu, nd_cpu, benchmark)
    proxy = 1.0 * wl + 0.5 * den + 0.5 * cong
    return {
        "proxy_cost":      proxy,
        "wirelength_cost": wl,
        "density_cost":    den,
        "congestion_cost": cong,
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate(pos: torch.Tensor, net_data: dict, benchmark: Benchmark, plc) -> None:
    """
    Compare our independent proxy against the harness PlacementCost.

    Prints a side-by-side table.  Call after placement to calibrate the
    deviation between our approximation and ground truth.
    """
    from macro_place.objective import compute_proxy_cost
    harness = compute_proxy_cost(pos, benchmark, plc)
    ours    = compute_proxy(pos, net_data, benchmark)

    rows = [
        ("wl_cost",         ours["wirelength_cost"], harness["wirelength_cost"]),
        ("density_cost",    ours["density_cost"],    harness["density_cost"]),
        ("congestion_cost", ours["congestion_cost"], harness["congestion_cost"]),
        ("proxy_cost",      ours["proxy_cost"],      harness["proxy_cost"]),
    ]
    print("  ── proxy validation ─────────────────────────────────────────────")
    print(f"  {'term':<18s}  {'ours':>9s}  {'harness':>9s}  {'err%':>7s}")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*7}")
    for name, ours_val, ref_val in rows:
        err = (ours_val - ref_val) / (abs(ref_val) + 1e-12) * 100
        print(f"  {name:<18s}  {ours_val:9.4f}  {ref_val:9.4f}  {err:+7.2f}%")
    print("  ─────────────────────────────────────────────────────────────────")
