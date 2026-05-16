"""
GACometPlacer — Genetic algorithm over CometPlacer hyperparameters.

Wraps CometPlacer.place() as a fitness function and evolves a population of
configs over multiple generations, returning the best placement found.

Enable via config.toml:
    [ga]
    ga_enable  = true
    pop_size   = 16
    n_gens     = 5
    elite_k    = 2
    tourn_k    = 3
    mut_rate   = 0.25

Gene space is defined in GENE_SPACE below. Each entry is one of:
    ("linear", min, max)      — Gaussian mutation, arithmetic crossover
    ("log",    min, max)      — Gaussian mutation in log-space
    ("choice", [v1, v2, ...]) — uniform crossover, random-choice mutation
"""

import copy
import math
import random
import sys
import time
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Gene space — edit to add/remove parameters
# ---------------------------------------------------------------------------

GENE_SPACE: dict[str, tuple] = {
    "lambda_den_init":               ("log",    1e-3,  3e-1),
    "halo_size":                     ("linear", 0.0,   0.5),
    "max_step":                      ("linear", 0.002, 0.010),
    "gamma_decay":                   ("linear", 0.988, 0.998),
    "target_density":                ("linear", 0.60,  0.80),
    "mGP_hard_macro_density_weight": ("linear", 0.8,   2.0),
    "mGP_soft_macro_density_weight": ("linear", 0.5,   1.2),
    "initial_placement":             ("choice", ["none", "center", "quadratic"]),
    "rotation_optimizer":            ("choice", ["none", "greedy"]),
}

# ---------------------------------------------------------------------------
# GA defaults (overridden by config.toml [ga] section)
# ---------------------------------------------------------------------------

GA_DEFAULTS = {
    "pop_size":  16,
    "n_gens":    5,
    "elite_k":   2,
    "tourn_k":   3,
    "mut_rate":  0.25,
    "seed":      0,
}

OVERLAP_PENALTY = 10.0   # added to proxy when overlaps > 0


# ---------------------------------------------------------------------------
# Individual representation
# ---------------------------------------------------------------------------

def _sample(gene_spec: tuple) -> float | str:
    kind = gene_spec[0]
    if kind == "linear":
        _, lo, hi = gene_spec
        return random.uniform(lo, hi)
    elif kind == "log":
        _, lo, hi = gene_spec
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    else:  # choice
        return random.choice(gene_spec[1])


def _clamp(val, gene_spec: tuple):
    kind = gene_spec[0]
    if kind in ("linear", "log"):
        return max(gene_spec[1], min(gene_spec[2], val))
    return val


def sample_individual() -> dict:
    return {k: _sample(spec) for k, spec in GENE_SPACE.items()}


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover(a: dict, b: dict) -> dict:
    child = {}
    for k, spec in GENE_SPACE.items():
        kind = spec[0]
        if kind == "choice":
            child[k] = random.choice([a[k], b[k]])
        elif kind == "linear":
            alpha = random.uniform(-0.25, 1.25)   # BLX-style, allows extrapolation
            child[k] = _clamp(alpha * a[k] + (1 - alpha) * b[k], spec)
        else:  # log
            _, lo, hi = spec
            la, lb = math.log(a[k]), math.log(b[k])
            alpha = random.uniform(-0.25, 1.25)
            child[k] = _clamp(math.exp(alpha * la + (1 - alpha) * lb), spec)
    return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(ind: dict, rate: float) -> dict:
    out = {}
    for k, spec in GENE_SPACE.items():
        if random.random() > rate:
            out[k] = ind[k]
            continue
        kind = spec[0]
        if kind == "choice":
            out[k] = random.choice(spec[1])
        elif kind == "linear":
            _, lo, hi = spec
            sigma = (hi - lo) * 0.1
            out[k] = _clamp(ind[k] + random.gauss(0, sigma), spec)
        else:  # log
            _, lo, hi = spec
            sigma = (math.log(hi) - math.log(lo)) * 0.1
            out[k] = _clamp(math.exp(math.log(ind[k]) + random.gauss(0, sigma)), spec)
    return out


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def tournament(population: list[dict], fitnesses: list[float], k: int) -> dict:
    contestants = random.sample(range(len(population)), k)
    winner = min(contestants, key=lambda i: fitnesses[i])
    return population[winner]


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _evaluate(ind: dict, benchmark, base_config: dict, run_idx: int) -> tuple[float, dict]:
    """Run one CometPlacer with ind's genes injected; return (fitness, costs)."""
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("params", {}).update(ind)
    cfg.setdefault("output", {}).update({"record_frames": False, "quiet": True})
    cfg.setdefault("ga", {})["ga_enable"] = False   # prevent recursive GA instantiation

    placer_mod = _import_placer()
    placer = placer_mod.CometPlacer(config=cfg)

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
    bm_fresh, plc = load_benchmark_from_dir(str(root))

    t0 = time.perf_counter()
    placement = placer.place(bm_fresh)
    elapsed = time.perf_counter() - t0

    _, violations = validate_placement(placement, bm_fresh)
    costs = compute_proxy_cost(placement, bm_fresh, plc)
    overlaps = costs["overlap_count"]
    proxy    = costs["proxy_cost"]
    fitness  = proxy + (OVERLAP_PENALTY if overlaps > 0 else 0.0)

    return fitness, {**costs, "elapsed": elapsed, "placement": placement,
                     "macro_sizes": bm_fresh.macro_sizes.clone()}


# ---------------------------------------------------------------------------
# GACometPlacer
# ---------------------------------------------------------------------------

class GACometPlacer:
    """
    Genetic algorithm wrapper around CometPlacer.

    place(benchmark) runs the GA and returns the placement with the best
    fitness (lowest proxy + overlap penalty) found across all generations.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = _load_config()
        self._base_config = config
        ga = config.get("ga", {})
        self.pop_size  = int(ga.get("pop_size",  GA_DEFAULTS["pop_size"]))
        self.n_gens    = int(ga.get("n_gens",    GA_DEFAULTS["n_gens"]))
        self.elite_k   = int(ga.get("elite_k",   GA_DEFAULTS["elite_k"]))
        self.tourn_k   = int(ga.get("tourn_k",   GA_DEFAULTS["tourn_k"]))
        self.mut_rate  = float(ga.get("mut_rate", GA_DEFAULTS["mut_rate"]))
        self.rng_seed  = int(ga.get("seed",       GA_DEFAULTS["seed"]))

    def place(self, benchmark):
        random.seed(self.rng_seed)

        population = [sample_individual() for _ in range(self.pop_size)]
        fitnesses  = [float("inf")] * self.pop_size

        best_fitness  = float("inf")
        best_pos      = None
        best_costs    = None
        best_genes    = None
        run_idx       = 0

        t_total = time.perf_counter()

        for gen in range(self.n_gens):
            t_gen = time.perf_counter()
            print(f"\n{'─'*60}")
            print(f"  [GA] Generation {gen+1}/{self.n_gens}  "
                  f"(pop={self.pop_size})")
            print(f"{'─'*60}")

            for i, ind in enumerate(population):
                run_idx += 1
                genes_str = "  ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in ind.items()
                )
                print(f"  [GA] run {run_idx:3d} | ind {i+1:2d}/{self.pop_size} | {genes_str}")

                try:
                    fitness, costs = _evaluate(
                        ind, benchmark, self._base_config, run_idx
                    )
                except Exception as exc:
                    print(f"  [GA] run {run_idx} FAILED: {exc}", file=sys.stderr)
                    fitness = float("inf")
                    costs   = {}

                fitnesses[i] = fitness
                tag = "  ← best!" if fitness < best_fitness else ""
                print(
                    f"  [GA] run {run_idx:3d} → fitness={fitness:.4f}  "
                    f"proxy={costs.get('proxy_cost', float('nan')):.4f}  "
                    f"overlaps={costs.get('overlap_count', '?')}  "
                    f"({costs.get('elapsed', 0):.1f}s){tag}"
                )

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_pos     = costs.get("placement")
                    best_costs   = costs
                    best_genes   = ind

            # Generation summary
            valid_fits = [f for f in fitnesses if f < float("inf")]
            avg_fit = sum(valid_fits) / len(valid_fits) if valid_fits else float("nan")
            print(
                f"\n  [GA] Gen {gen+1} done | "
                f"best={min(fitnesses):.4f}  avg={avg_fit:.4f}  "
                f"time={time.perf_counter()-t_gen:.1f}s"
            )

            if gen == self.n_gens - 1:
                break

            # Evolve: elitism + tournament + crossover + mutation
            ranked = sorted(range(self.pop_size), key=lambda i: fitnesses[i])
            new_pop = [population[i] for i in ranked[:self.elite_k]]

            while len(new_pop) < self.pop_size:
                p1 = tournament(population, fitnesses, self.tourn_k)
                p2 = tournament(population, fitnesses, self.tourn_k)
                child = crossover(p1, p2)
                child = mutate(child, self.mut_rate)
                new_pop.append(child)

            population = new_pop
            fitnesses  = [float("inf")] * self.pop_size

        print(
            f"\n  [GA] Done — {run_idx} total runs  "
            f"best_fitness={best_fitness:.4f}  "
            f"total_time={time.perf_counter()-t_total:.1f}s"
        )
        if best_costs:
            print(
                f"  [GA] Best proxy={best_costs.get('proxy_cost'):.4f}  "
                f"overlaps={best_costs.get('overlap_count')}"
            )
        if best_genes:
            print("\n  [GA] Best genes:")
            for k, v in best_genes.items():
                print(f"    {k:<36} = {v:.6g}" if isinstance(v, float) else f"    {k:<36} = {v}")

        self._write_results(benchmark.name, run_idx,
                            time.perf_counter() - t_total,
                            best_fitness, best_costs, best_genes)

        return best_pos


    def _write_results(self, bench_name, total_runs, elapsed,
                       best_fitness, best_costs, best_genes):
        frames_dir = Path(
            self._base_config.get("output", {}).get("frames_dir", "vis/frames")
        ) / bench_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        out = frames_dir / "ga_results.txt"

        lines = [
            f"benchmark   : {bench_name}",
            f"total_runs  : {total_runs}",
            f"total_time  : {elapsed:.1f}s",
            f"pop_size    : {self.pop_size}",
            f"n_gens      : {self.n_gens}",
            "",
        ]
        if best_costs:
            lines += [
                f"best_fitness  : {best_fitness:.6f}",
                f"best_proxy    : {best_costs.get('proxy_cost', float('nan')):.6f}",
                f"best_wl       : {best_costs.get('wirelength_cost', float('nan')):.6f}",
                f"best_density  : {best_costs.get('density_cost', float('nan')):.6f}",
                f"best_cong     : {best_costs.get('congestion_cost', float('nan')):.6f}",
                f"overlaps      : {best_costs.get('overlap_count', '?')}",
                "",
            ]
        if best_genes:
            lines.append("best_genes:")
            for k, v in best_genes.items():
                val = f"{v:.8g}" if isinstance(v, float) else str(v)
                lines.append(f"  {k:<36} = {val}")

        out.write_text("\n".join(lines) + "\n")
        print(f"  [GA] Results written to {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_placer():
    spec = importlib.util.spec_from_file_location(
        "placer", Path(__file__).parent / "placer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_config() -> dict:
    import tomllib
    env_path = __import__("os").environ.get("MSPLACER_CONFIG")
    path = Path(env_path) if env_path else Path(__file__).parent / "config.toml"
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
