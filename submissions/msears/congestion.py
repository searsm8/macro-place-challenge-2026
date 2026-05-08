"""
RUDY (Rectangular Uniform Wire DensitY) congestion estimator.

For each net k with bounding box [x1,x2]×[y1,y2] and HPWL h_k:
  routing demand density = h_k / bbox_area_k   (wires per unit area)

This density is spread uniformly over all grid cells that the bbox covers,
weighted by their fractional overlap with the bbox. The result is a 2D
congestion map where high values indicate routing hotspots.

Reference: Spinder, 2007. "Fast and Accurate Routing Demand Estimation 
    for Efficient Routability-driven Placement", aka RUDY
"""

import torch


def compute_rudy_map(
    pos: torch.Tensor,
    net_data: dict,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    """
    Vectorized RUDY congestion map.

    Args:
        pos:        [N, 2] macro center positions (CPU or CUDA)
        net_data:   dict with macro_ids, offsets, net_ids, is_macro, num_nets
        canvas_w/h: chip canvas dimensions
        grid_rows/cols: congestion grid resolution

    Returns:
        congestion_map [grid_rows, grid_cols]  — routing demand per unit area per cell
    """
    device = pos.device
    K = net_data["num_nets"]

    macro_ids = net_data["macro_ids"]
    offsets   = net_data["offsets"]
    net_ids_t = net_data["net_ids"]
    is_macro  = net_data["is_macro"]

    safe_ids = macro_ids.clamp(min=0)
    mflag    = is_macro.float().unsqueeze(1)
    pin_pos  = pos[safe_ids] * mflag + offsets  # [P, 2]

    # Per-net bounding box via scatter
    big = torch.full((K,), float("inf"),  device=device)
    neg = torch.full((K,), float("-inf"), device=device)
    x_min = big.clone().scatter_reduce_(0, net_ids_t, pin_pos[:, 0], reduce="amin", include_self=True)
    x_max = neg.clone().scatter_reduce_(0, net_ids_t, pin_pos[:, 0], reduce="amax", include_self=True)
    y_min = big.clone().scatter_reduce_(0, net_ids_t, pin_pos[:, 1], reduce="amin", include_self=True)
    y_max = neg.clone().scatter_reduce_(0, net_ids_t, pin_pos[:, 1], reduce="amax", include_self=True)

    bbox_w = (x_max - x_min).clamp(min=1e-6)
    bbox_h = (y_max - y_min).clamp(min=1e-6)
    hpwl_k = bbox_w + bbox_h
    demand  = hpwl_k / (bbox_w * bbox_h)  # [K]  wires per unit area

    cell_w    = canvas_w / grid_cols
    cell_h    = canvas_h / grid_rows
    cell_area = cell_w * cell_h

    col_edges = torch.linspace(0.0, canvas_w, grid_cols + 1, device=device)  # [C+1]
    row_edges = torch.linspace(0.0, canvas_h, grid_rows + 1, device=device)  # [R+1]

    # overlap_x[k, c] = max(0, min(x_max[k], col_edges[c+1]) - max(x_min[k], col_edges[c]))
    overlap_x = (
        torch.minimum(x_max.unsqueeze(1), col_edges[1:].unsqueeze(0))
        - torch.maximum(x_min.unsqueeze(1), col_edges[:-1].unsqueeze(0))
    ).clamp(min=0.0)  # [K, C]

    overlap_y = (
        torch.minimum(y_max.unsqueeze(1), row_edges[1:].unsqueeze(0))
        - torch.maximum(y_min.unsqueeze(1), row_edges[:-1].unsqueeze(0))
    ).clamp(min=0.0)  # [K, R]

    # congestion[r,c] = (1/cell_area) * Σ_k demand[k] * overlap_y[k,r] * overlap_x[k,c]
    #                 = overlap_y.T @ (demand[:,None] * overlap_x) / cell_area
    weighted_x    = demand.unsqueeze(1) * overlap_x          # [K, C]
    congestion_map = (overlap_y.T @ weighted_x) / cell_area  # [R, C]

    return congestion_map


def rudy_stats(congestion_map: torch.Tensor) -> dict:
    """
    Summary statistics from a RUDY congestion map.

    Returns:
        mean, max, and p99 (99th-percentile) routing demand.
    """
    flat = congestion_map.flatten()
    p99  = torch.quantile(flat.float(), 0.99).item()
    return {
        "rudy_mean": flat.mean().item(),
        "rudy_max":  flat.max().item(),
        "rudy_p99":  p99,
    }
