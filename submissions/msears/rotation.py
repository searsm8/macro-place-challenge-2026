"""
rotation.py — Post-placement and mid-placement orientation optimizer.

Implements greedy and simulated-annealing (SA) rotators that improve
hard-macro orientations by minimising HPWL at fixed positions.

  greedy_rotate  — single pass, always accepts improvements
  sa_rotate      — annealed search, accepts uphill moves with probability e^(-Δ/T)

Both are called from placer.py:
  - "greedy" / "anneal"  → after legalization, before soft_place
  - "periodic"           → every rotation_period iters inside the gradient loop
                           (greedy only; SA needs many steps to be meaningful)

Interface
---------
All functions take:
  pos          FloatTensor [n, 2]   current (legalized) macro positions
  work_offs    FloatTensor [P, 2]   mutable offsets — modified in-place on return
  net_data_base dict                net_data with all-N pin offsets (never modified)
  genes        int8 Tensor [num_hard]  current orientation per hard macro (modified)
  benchmark    Benchmark object
  gamma        float                current WA gamma (used for WL evaluation)

Return value: n_flipped (int), for logging.
"""

from __future__ import annotations
import math
import torch

# ---------------------------------------------------------------------------
# Orientation matrices — D4 group (indices 0–7)
# ---------------------------------------------------------------------------
# Row i = [a, b, c, d]:  dx' = a*dx + b*dy,  dy' = c*dx + d*dy
_ORI_MATRICES = torch.tensor(
    [
        [ 1,  0,  0,  1],   # 0 N
        [-1,  0,  0,  1],   # 1 FN
        [-1,  0,  0, -1],   # 2 S
        [ 1,  0,  0, -1],   # 3 FS
        [ 0, -1,  1,  0],   # 4 E
        [ 0,  1,  1,  0],   # 5 FE
        [ 0,  1, -1,  0],   # 6 W
        [ 0, -1, -1,  0],   # 7 FW
    ],
    dtype=torch.float32,
)  # [8, 4]


# ---------------------------------------------------------------------------
# HPWL for a subset of pins
# ---------------------------------------------------------------------------

def _hpwl_subnet(pos, offsets, macro_ids, net_ids, is_macro, pin_mask):
    """
    Exact HPWL restricted to the pins selected by pin_mask.

    Uses scatter_reduce_ amax/amin — no autograd, purely for scoring.
    """
    m   = macro_ids[pin_mask].clamp(min=0)
    off = offsets[pin_mask]
    nid = net_ids[pin_mask]
    ism = is_macro[pin_mask]

    pp = pos[m].detach() * ism.float().unsqueeze(-1) + off   # [P', 2]
    _, nc = torch.unique(nid, return_inverse=True)            # nc in [0, K)
    K  = int(nc.max().item()) + 1

    NEG = torch.full((K,), float("-inf"))
    POS = torch.full((K,),  float("inf"))
    xmax = NEG.clone().scatter_reduce_(0, nc, pp[:, 0], reduce="amax")
    xmin = POS.clone().scatter_reduce_(0, nc, pp[:, 0], reduce="amin")
    ymax = NEG.clone().scatter_reduce_(0, nc, pp[:, 1], reduce="amax")
    ymin = POS.clone().scatter_reduce_(0, nc, pp[:, 1], reduce="amin")
    return float(((xmax - xmin) + (ymax - ymin)).sum())


# ---------------------------------------------------------------------------
# Shared setup: precompute per-macro pin-masks and net-pin-masks
# ---------------------------------------------------------------------------

def _build_macro_masks(net_data_base, num_hard):
    """
    Returns two lists, each of length num_hard:
      m_masks      [P] bool  — pins that belong to macro m
      net_pin_masks [P] bool  — all pins in any net touching macro m
                               (used to restrict HPWL to only relevant nets)
    Returns None for macros with no pins.
    """
    macro_ids = net_data_base["macro_ids"]
    net_ids   = net_data_base["net_ids"]

    m_masks: list       = []
    net_pin_masks: list = []

    for m in range(num_hard):
        mm = (macro_ids == m)
        if not mm.any():
            m_masks.append(None)
            net_pin_masks.append(None)
            continue
        m_nets = torch.unique(net_ids[mm])
        m_masks.append(mm)
        net_pin_masks.append(torch.isin(net_ids, m_nets))

    return m_masks, net_pin_masks


# ---------------------------------------------------------------------------
# Apply an orientation to one macro's pins in the working offsets buffer
# ---------------------------------------------------------------------------

def _apply_ori(work_offs, base_offs, m_mask, ori):
    mat = _ORI_MATRICES[ori]
    dx  = base_offs[m_mask, 0]
    dy  = base_offs[m_mask, 1]
    work_offs[m_mask, 0] = mat[0] * dx + mat[1] * dy
    work_offs[m_mask, 1] = mat[2] * dx + mat[3] * dy


# ---------------------------------------------------------------------------
# Compute per-macro HPWL contributions for sorting
# ---------------------------------------------------------------------------

def _per_macro_hpwl(pos, work_offs, net_data_base, m_masks, net_pin_masks, num_hard):
    macro_ids = net_data_base["macro_ids"]
    net_ids   = net_data_base["net_ids"]
    is_macro  = net_data_base["is_macro"]
    contribs  = torch.zeros(num_hard)
    for m in range(num_hard):
        if net_pin_masks[m] is None:
            continue
        contribs[m] = _hpwl_subnet(
            pos, work_offs, macro_ids, net_ids, is_macro, net_pin_masks[m]
        )
    return contribs


# ---------------------------------------------------------------------------
# Greedy rotator
# ---------------------------------------------------------------------------

def greedy_rotate(pos, work_offs, net_data_base, genes, benchmark, gamma=None,
                  n_passes=1, candidates=(0, 1, 2, 3, 4, 5, 6, 7)):
    """
    For each hard macro (sorted by descending HPWL contribution), try the
    candidate orientations and commit the one with lowest HPWL on connected nets.

    candidates: tuple of orientation indices to consider (default all 8).
      Use (0, 2) for N/S-only (safest plc-sync, no size swap).
      Use (0, 1, 2, 3) to also include mirror orientations.

    Modifies work_offs and genes in-place.
    Returns n_flipped.
    """
    num_hard  = benchmark.num_hard_macros
    macro_ids = net_data_base["macro_ids"]
    net_ids   = net_data_base["net_ids"]
    is_macro  = net_data_base["is_macro"]
    base_offs = net_data_base["offsets"]

    # Clone base_offs so writes to work_offs never corrupt it.
    # Without this, if the caller passes net_data["offsets"] and
    # net_data_base["offsets"] pointing to the same tensor (which
    # happens when GA is disabled and net_data_base = net_data),
    # the first committed flip would corrupt all subsequent lookups.
    base_offs = base_offs.clone()

    m_masks, net_pin_masks = _build_macro_masks(net_data_base, num_hard)
    n_flipped = 0

    for _ in range(n_passes):
        # Sort by current per-macro HPWL contribution (biggest mover first)
        contribs = _per_macro_hpwl(
            pos, work_offs, net_data_base, m_masks, net_pin_masks, num_hard
        )
        order = contribs.argsort(descending=True).tolist()

        for m in order:
            if m_masks[m] is None:
                continue
            pm = net_pin_masks[m]
            cur_wl = _hpwl_subnet(pos, work_offs, macro_ids, net_ids, is_macro, pm)
            best_wl  = cur_wl
            best_ori = int(genes[m])

            for ori in candidates:
                if ori == int(genes[m]):
                    continue
                _apply_ori(work_offs, base_offs, m_masks[m], ori)
                wl = _hpwl_subnet(pos, work_offs, macro_ids, net_ids, is_macro, pm)
                if wl < best_wl:
                    best_wl  = wl
                    best_ori = ori
                # Restore current orientation before next trial
                _apply_ori(work_offs, base_offs, m_masks[m], int(genes[m]))

            if best_ori != int(genes[m]):
                _apply_ori(work_offs, base_offs, m_masks[m], best_ori)
                genes[m] = best_ori
                n_flipped += 1

    return n_flipped


# ---------------------------------------------------------------------------
# Simulated annealing rotator
# ---------------------------------------------------------------------------

def sa_rotate(pos, work_offs, net_data_base, genes, benchmark, gamma=None,
              n_steps=None, T_init=1e-4, T_final=1e-7, seed=0,
              candidates=(0, 1, 2, 3, 4, 5, 6, 7)):
    """
    SA orientation search. Each step:
      1. Pick a random hard macro m
      2. Pick a random orientation o ≠ current
      3. Compute ΔHPWL on the nets touching m
      4. Accept if ΔHPWL < 0 or with probability exp(-ΔHPWL / T)

    n_steps defaults to 200 × num_hard if not specified.
    Modifies work_offs and genes in-place.
    Returns n_accepted.
    """
    num_hard  = benchmark.num_hard_macros
    macro_ids = net_data_base["macro_ids"]
    net_ids   = net_data_base["net_ids"]
    is_macro  = net_data_base["is_macro"]
    base_offs = net_data_base["offsets"]

    if n_steps is None:
        n_steps = 200 * num_hard

    base_offs = base_offs.clone()  # guard against aliasing with work_offs

    m_masks, net_pin_masks = _build_macro_masks(net_data_base, num_hard)
    valid_macros = [m for m in range(num_hard) if m_masks[m] is not None]

    rng = torch.Generator()
    rng.manual_seed(seed)
    n_accepted = 0

    for step in range(n_steps):
        # Temperature: exponential decay
        frac = step / max(n_steps - 1, 1)
        T = T_init * (T_final / T_init) ** frac

        # Pick random macro and a random candidate orientation != current
        m   = valid_macros[int(torch.randint(len(valid_macros), (1,), generator=rng))]
        cur = int(genes[m])
        other = [o for o in candidates if o != cur]
        if not other:
            continue
        ori = other[int(torch.randint(len(other), (1,), generator=rng))]

        pm = net_pin_masks[m]
        cur_wl = _hpwl_subnet(pos, work_offs, macro_ids, net_ids, is_macro, pm)

        _apply_ori(work_offs, base_offs, m_masks[m], ori)
        new_wl = _hpwl_subnet(pos, work_offs, macro_ids, net_ids, is_macro, pm)

        delta_wl = new_wl - cur_wl
        accept   = delta_wl < 0 or (
            T > 0 and float(torch.rand(1, generator=rng)) < math.exp(-delta_wl / T)
        )

        if accept:
            genes[m] = ori
            n_accepted += 1
        else:
            # Revert
            _apply_ori(work_offs, base_offs, m_masks[m], cur)

    return n_accepted
