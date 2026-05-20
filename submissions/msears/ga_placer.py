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
    "halo_size":                     ("linear", 0.10,  0.25),
    "max_step":                      ("linear", 0.003, 0.007),
    "target_density":                ("linear", 0.60,  0.80),
    "mGP_hard_macro_density_weight": ("linear", 1.0,   1.4),
    "lambda_den_init":               ("log",    8e-6,  1e-3),
    "lambda_cong_target":            ("linear", 0.25,  0.38),
    "quad_scatter_lock_mult":        ("log",    1e5,   3e6),
}

# ---------------------------------------------------------------------------
# GA defaults (overridden by config.toml [ga] section)
# ---------------------------------------------------------------------------

GA_DEFAULTS = {
    "pop_size":  16,
    "n_gens":    4,
    "elite_k":   2,
    "tourn_k":   3,
    "mut_rate":  0.15,
    "seed":      0,
}

OVERLAP_PENALTY = 10.0   # added to proxy when overlaps > 0


# ---------------------------------------------------------------------------
# Individual representation
# ---------------------------------------------------------------------------

def _sample(gene_spec: tuple) -> float | int | str:
    kind = gene_spec[0]
    if kind == "linear":
        _, lo, hi = gene_spec
        return random.uniform(lo, hi)
    elif kind == "log":
        _, lo, hi = gene_spec
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    elif kind == "int":
        _, lo, hi = gene_spec
        return random.randint(int(lo), int(hi))
    else:  # choice
        return random.choice(gene_spec[1])


def _clamp(val, gene_spec: tuple):
    kind = gene_spec[0]
    if kind == "int":
        return int(max(gene_spec[1], min(gene_spec[2], round(val))))
    if kind in ("linear", "log"):
        return max(gene_spec[1], min(gene_spec[2], val))
    return val


def sample_individual() -> dict:
    return {k: _sample(spec) for k, spec in GENE_SPACE.items()}


# Keys whose config-file values live in [congestion] rather than [params]
_CONGESTION_KEYS = {"lambda_cong_init", "lambda_cong_ramp", "lambda_cong_target"}

def default_individual(base_config: dict) -> dict:
    """Return a gene dict populated from base_config values (clamped to gene ranges)."""
    params = base_config.get("params", {})
    cong   = base_config.get("congestion", {})
    ind = {}
    for k, spec in GENE_SPACE.items():
        section = cong if k in _CONGESTION_KEYS else params
        val = section.get(k)
        if val is None:
            val = _sample(spec)   # gene not in config — fall back to random
        ind[k] = _clamp(val, spec)
    return ind


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover(a: dict, b: dict) -> dict:
    child = {}
    for k, spec in GENE_SPACE.items():
        kind = spec[0]
        if kind == "choice":
            child[k] = random.choice([a[k], b[k]])
        elif kind == "int":
            alpha = random.uniform(-0.25, 1.25)
            child[k] = _clamp(alpha * a[k] + (1 - alpha) * b[k], spec)
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
        elif kind == "int":
            _, lo, hi = spec
            sigma = max(1, (hi - lo) * 0.1)
            out[k] = _clamp(ind[k] + random.gauss(0, sigma), spec)
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
    params_genes = {k: v for k, v in ind.items() if k not in _CONGESTION_KEYS}
    cong_genes   = {k: v for k, v in ind.items() if k in _CONGESTION_KEYS}
    cfg.setdefault("params", {}).update(params_genes)
    cfg.setdefault("congestion", {}).update(cong_genes)
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
        n_sec = int(config.get("params", {}).get("quad_scatter_section_count", 4))
        GENE_SPACE["largest_macro_starting_section"] = ("int", 1, max(1, n_sec))
        ga = config.get("ga", {})
        self.pop_size  = int(ga.get("pop_size",  GA_DEFAULTS["pop_size"]))
        self.n_gens    = int(ga.get("n_gens",    GA_DEFAULTS["n_gens"]))
        self.elite_k   = int(ga.get("elite_k",   GA_DEFAULTS["elite_k"]))
        self.tourn_k   = int(ga.get("tourn_k",   GA_DEFAULTS["tourn_k"]))
        self.mut_rate   = float(ga.get("mut_rate",       GA_DEFAULTS["mut_rate"]))
        self.rng_seed   = int(ga.get("seed",             GA_DEFAULTS["seed"]))
        self.time_limit = float(ga.get("time_limit_mins", 55)) * 60
        self.final_seeds = list(ga.get("final_seeds", []))
        raw = ga.get("perform_seed_sweep", False)
        self.perform_seed_sweep = raw if isinstance(raw, bool) else str(raw).lower() == "true"

    def place(self, benchmark):
        random.seed(self.rng_seed)

        history_path = self._history_path(benchmark.name)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text("")  # truncate any prior history

        population = [default_individual(self._base_config)] + [
            sample_individual() for _ in range(self.pop_size - 1)
        ]
        fitnesses  = [float("inf")] * self.pop_size

        best_fitness  = float("inf")
        best_pos      = None
        best_costs    = None
        best_genes    = None
        run_idx       = 0
        run_times     = []
        time_limit    = self.time_limit
        timed_out     = False

        t_total = time.perf_counter()

        for gen in range(self.n_gens):
            t_gen = time.perf_counter()
            print(f"\n{'─'*60}")
            print(f"  [GA] Generation {gen+1}/{self.n_gens}  "
                  f"(pop={self.pop_size})")
            print(f"{'─'*60}")

            for i, ind in enumerate(population):
                elapsed_so_far = time.perf_counter() - t_total
                avg_run = sum(run_times) / len(run_times) if run_times else 0.0
                if run_times and elapsed_so_far + avg_run > time_limit:
                    print(
                        f"  [GA] Time limit: {elapsed_so_far/60:.1f}m elapsed + "
                        f"{avg_run:.0f}s avg run would exceed {time_limit/60:.0f}m limit — stopping early"
                    )
                    timed_out = True
                    break

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
                    run_times.append(costs.get("elapsed", 0.0))
                except Exception as exc:
                    print(f"  [GA] run {run_idx} FAILED: {exc}", file=sys.stderr)
                    fitness = float("inf")
                    costs   = {}

                fitnesses[i] = fitness
                self._append_history(history_path, gen, i, run_idx, ind, fitness, costs)
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

            if timed_out or gen == self.n_gens - 1:
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

        # Final seed sweep over the best gene set found by GA
        if best_genes and self.final_seeds and self.perform_seed_sweep:
            print(f"\n{'─'*60}")
            print(f"  [GA] Final seed sweep — best_genes × {len(self.final_seeds)} seeds")
            print(f"{'─'*60}")
            for s in self.final_seeds:
                elapsed_so_far = time.perf_counter() - t_total
                avg_run = sum(run_times) / len(run_times) if run_times else 0.0
                if run_times and elapsed_so_far + avg_run > time_limit:
                    print(
                        f"  [seed-sweep] Time limit: {elapsed_so_far/60:.1f}m elapsed + "
                        f"{avg_run:.0f}s avg run would exceed {time_limit/60:.0f}m limit — stopping early"
                    )
                    break
                run_idx += 1
                ind = {**best_genes, "seed": s}
                print(f"  [seed-sweep] run {run_idx:3d} | seed={s}")
                try:
                    fitness, costs = _evaluate(ind, benchmark, self._base_config, run_idx)
                except Exception as exc:
                    print(f"  [seed-sweep] run {run_idx} FAILED: {exc}", file=sys.stderr)
                    fitness, costs = float("inf"), {}
                self._append_history(history_path, -1, s, run_idx, ind, fitness, costs)
                tag = "  ← best!" if fitness < best_fitness else ""
                print(
                    f"  [seed-sweep] run {run_idx:3d} → fitness={fitness:.4f}  "
                    f"proxy={costs.get('proxy_cost', float('nan')):.4f}  "
                    f"overlaps={costs.get('overlap_count', '?')}  "
                    f"({costs.get('elapsed', 0):.1f}s){tag}"
                )
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_pos     = costs.get("placement")
                    best_costs   = costs
                    best_genes   = ind

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


    def _out_dir(self, bench_name: str) -> Path:
        import os
        env_config = os.environ.get("MSPLACER_CONFIG")
        if env_config:
            return Path(env_config).parent
        return Path(
            self._base_config.get("output", {}).get("frames_dir", "vis/frames")
        ) / bench_name

    def _history_path(self, bench_name: str) -> Path:
        return self._out_dir(bench_name) / "ga_history.jsonl"

    @staticmethod
    def _append_history(path: Path, gen: int, ind_idx: int, run_idx: int,
                        genes: dict, fitness: float, costs: dict) -> None:
        import json, math
        fit = fitness if math.isfinite(fitness) else None
        record = {
            "gen": gen,
            "ind": ind_idx,
            "run": run_idx,
            "fitness": fit,
            "proxy": costs.get("proxy_cost"),
            "wl": costs.get("wirelength_cost"),
            "density": costs.get("density_cost"),
            "cong": costs.get("congestion_cost"),
            "overlaps": costs.get("overlap_count"),
            "elapsed": costs.get("elapsed"),
            "genes": genes,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _write_results(self, bench_name, total_runs, elapsed,
                       best_fitness, best_costs, best_genes):
        out_dir = self._out_dir(bench_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "ga_results.txt"

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
