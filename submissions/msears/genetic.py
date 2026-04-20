"""
genetic.py — Genetic Algorithm outer loop for macro orientation optimization.

The GA searches over hard-macro orientations (one of 8 D4 symmetries per macro)
using curtailed placements as a fitness oracle, then hands the best chromosome
back to the full placer for a final high-quality run.

Typical call chain (inside CometPlacer.place):

    ga   = GeneticPlacer(ga_section_dict)
    best = ga.run(benchmark, net_data_base, eval_fn)
    net_data_mod, bench_mod = best.applyToPlacement(net_data_base, benchmark)
    # ... continue with full gradient descent on (bench_mod, net_data_mod)

eval_fn signature:
    eval_fn(benchmark_mod, net_data_mod) -> {"wl": float, "overflow": float}

────────────────────────────────────────────────────────────────────────────────
Orientation encoding
────────────────────────────────────────────────────────────────────────────────
Index  Name   Transform (dx,dy) → (dx',dy')   Size swap
  0     N     ( dx,  dy)                        no
  1     FN    (-dx,  dy)    mirror X            no
  2     S     (-dx, -dy)    rotate 180°         no
  3     FS    ( dx, -dy)    mirror Y            no
  4     E     (-dy,  dx)    rotate  90° CCW     yes (w↔h)
  5     FE    ( dy,  dx)    mirror X + E        yes (w↔h)
  6     W     ( dy, -dx)    rotate 270° CCW     yes (w↔h)
  7     FW    (-dy, -dx)    mirror X + W        yes (w↔h)

The 8 orientations are the dihedral group D4.  A single integer gene (0–7) per
hard macro is simpler and equally expressive as 3 binary genes for tournament-
selection / uniform-crossover at this population scale.
"""

from __future__ import annotations

import random as _stdlib_random

import torch


# ---------------------------------------------------------------------------
# Orientation constants
# ---------------------------------------------------------------------------

ORIENTATIONS: list[str] = ["N", "FN", "S", "FS", "E", "FE", "W", "FW"]
ORI_TO_IDX:   dict[str, int] = {o: i for i, o in enumerate(ORIENTATIONS)}

# Pin-offset rotation matrices: new_offset = M @ old_offset
# Stored as [8, 4] where row i = [a, b, c, d]:
#   dx' = a*dx + b*dy
#   dy' = c*dx + d*dy
_ORI_MATRICES = torch.tensor(
    [
        [ 1,  0,  0,  1],   # 0: N   identity
        [-1,  0,  0,  1],   # 1: FN  mirror X
        [-1,  0,  0, -1],   # 2: S   rotate 180°
        [ 1,  0,  0, -1],   # 3: FS  mirror Y
        [ 0, -1,  1,  0],   # 4: E   rotate  90° CCW
        [ 0,  1,  1,  0],   # 5: FE  mirror X + rotate  90° CCW
        [ 0,  1, -1,  0],   # 6: W   rotate 270° CCW
        [ 0, -1, -1,  0],   # 7: FW  mirror X + rotate 270° CCW
    ],
    dtype=torch.float32,
)  # [8, 4]

# True for orientations that swap macro width ↔ height (E/FE/W/FW)
_ORI_SWAPS_SIZE = torch.tensor(
    [False, False, False, False, True, True, True, True]
)  # [8]


# ---------------------------------------------------------------------------
# Chromosome
# ---------------------------------------------------------------------------

class Chromosome:
    """
    Genome for one placement candidate.

    Attributes
    ----------
    genes : int8 tensor [num_hard]
        Orientation index 0–7 for each hard macro.  Soft macros always use
        the identity (N = 0) and are not encoded here.
    """

    __slots__ = ("genes",)

    def __init__(self, genes: torch.Tensor) -> None:
        self.genes = genes.to(dtype=torch.int8)

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(
        cls,
        num_hard: int,
        rng: torch.Generator | None = None,
    ) -> "Chromosome":
        """Sample each gene uniformly from {0 … 7}."""
        genes = torch.randint(0, 8, (num_hard,), generator=rng, dtype=torch.int8)
        return cls(genes)

    @classmethod
    def default(cls, num_hard: int) -> "Chromosome":
        """All-N orientation — identical to the benchmark's built-in default."""
        return cls(torch.zeros(num_hard, dtype=torch.int8))

    @classmethod
    def from_names(cls, orientations: list[str]) -> "Chromosome":
        """Build from a list of orientation strings, e.g. ['N', 'E', 'FN', …]."""
        genes = torch.tensor(
            [ORI_TO_IDX[o] for o in orientations], dtype=torch.int8
        )
        return cls(genes)

    def clone(self) -> "Chromosome":
        return Chromosome(self.genes.clone())

    # ------------------------------------------------------------------ #
    # Conversions                                                          #
    # ------------------------------------------------------------------ #

    def to_names(self) -> list[str]:
        return [ORIENTATIONS[i] for i in self.genes.tolist()]

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for name in self.to_names():
            counts[name] = counts.get(name, 0) + 1
        active = {k: v for k, v in counts.items() if v > 0}
        return f"Chromosome({active})"

    # ------------------------------------------------------------------ #
    # Genetic operators                                                    #
    # ------------------------------------------------------------------ #

    def uniform_crossover(
        self,
        other: "Chromosome",
        rng: torch.Generator | None = None,
    ) -> tuple["Chromosome", "Chromosome"]:
        """
        Uniform crossover: each gene is independently drawn from one parent
        or the other with equal probability.

        Produces two complementary children (mask and ~mask), so no genes
        are "lost" across the pair.
        """
        mask = torch.randint(
            0, 2, self.genes.shape, generator=rng, dtype=torch.bool
        )
        c1 = torch.where(mask, self.genes, other.genes).clone()
        c2 = torch.where(mask, other.genes, self.genes).clone()
        return Chromosome(c1), Chromosome(c2)

    def mutate(
        self,
        rate: float,
        rng: torch.Generator | None = None,
    ) -> "Chromosome":
        """
        Point mutation: each gene is replaced by a uniformly random orientation
        with probability *rate*.  The replacement is fully random (including the
        current value), keeping the distribution unbiased.
        """
        mask    = torch.rand(self.genes.shape, generator=rng) < rate
        new_val = torch.randint(
            0, 8, self.genes.shape, generator=rng, dtype=torch.int8
        )
        return Chromosome(torch.where(mask, new_val, self.genes))

    # ------------------------------------------------------------------ #
    # Apply to placement data                                              #
    # ------------------------------------------------------------------ #

    def applyToPlacement(
        self,
        net_data_base: dict,
        benchmark,
    ) -> tuple[dict, object]:
        """
        Transform pin offsets and macro sizes to match this chromosome.

        *net_data_base* must have been built with all macros in N orientation
        (straight from _buildNetData before any chromosome is applied).  The
        function is pure — it never modifies the inputs.

        Returns
        -------
        net_data_mod : dict
            New net_data with orientation-transformed offsets for macro pins.
            Port pins (is_macro == False) are unchanged.
        bench_mod : Benchmark
            Shallow copy of *benchmark* with corrected macro_sizes: E/W hard
            macros have width ↔ height swapped.  All other fields are shared
            references (read-only during placement, so this is safe).
        """
        import copy

        num_hard  = benchmark.num_hard_macros
        num_macros = benchmark.num_macros

        # ── 1. Transform pin offsets ─────────────────────────────────────
        # Orientation vector: hard macros → chromosome gene; soft macros → 0 (N)
        macro_ori = torch.zeros(num_macros, dtype=torch.long)
        macro_ori[:num_hard] = self.genes.long()

        # For each pin, inherit its macro's orientation (ports clamped to 0)
        safe_ids = net_data_base["macro_ids"].clamp(min=0)  # [P]
        pin_ori  = macro_ori[safe_ids]                      # [P]
        matrices = _ORI_MATRICES[pin_ori]                   # [P, 4]

        offsets  = net_data_base["offsets"]                 # [P, 2]
        is_macro = net_data_base["is_macro"]                # [P]

        dx, dy = offsets[:, 0], offsets[:, 1]
        new_dx = matrices[:, 0] * dx + matrices[:, 1] * dy
        new_dy = matrices[:, 2] * dx + matrices[:, 3] * dy
        new_offsets = torch.stack([new_dx, new_dy], dim=1)

        # Ports keep their absolute positions (is_macro == False)
        new_offsets = torch.where(is_macro.unsqueeze(1), new_offsets, offsets)
        net_data_mod = {**net_data_base, "offsets": new_offsets}

        # ── 2. Swap macro sizes for E / FE / W / FW orientations ────────
        swaps     = _ORI_SWAPS_SIZE[self.genes.long()]      # [num_hard] bool
        new_sizes = benchmark.macro_sizes.clone()           # [num_macros, 2]
        if swaps.any():
            new_sizes[:num_hard][swaps] = new_sizes[:num_hard][swaps].flip(1)

        # Shallow-copy the benchmark and replace only macro_sizes.
        # object.__setattr__ bypasses any frozen-dataclass guard; all other
        # attributes are shared references (they're read-only during placement).
        bench_mod = copy.copy(benchmark)
        object.__setattr__(bench_mod, "macro_sizes", new_sizes)
        return net_data_mod, bench_mod


# ---------------------------------------------------------------------------
# GA configuration
# ---------------------------------------------------------------------------

class GAConfig:
    """
    Hyperparameters for the outer GA loop, read from the ``[ga]`` section of
    config.toml (passed in as a plain dict).
    """

    def __init__(self, ga_section: dict) -> None:
        g = ga_section
        self.population_size     = g.get("population_size",          16)
        self.n_generations       = g.get("n_generations",            50)
        self.elite_frac          = g.get("elite_frac",             0.25)
        self.mutation_rate       = g.get("mutation_rate",          0.05)
        self.tournament_k        = g.get("tournament_k",              3)
        self.curtailed_iters     = g.get("curtailed_iters",         300)
        self.wl_weight           = g.get("fitness_wl_weight",       1.0)
        self.overflow_weight     = g.get("fitness_overflow_weight", 0.5)
        self.n_workers           = g.get("n_workers",                 1)
        self.seed                = g.get("seed",                      0)

    @property
    def n_elite(self) -> int:
        return max(1, int(self.elite_frac * self.population_size))

    def __repr__(self) -> str:
        return (
            f"GAConfig(pop={self.population_size}, gens={self.n_generations}, "
            f"elite={self.n_elite}, mut={self.mutation_rate}, "
            f"k={self.tournament_k}, curtailed_iters={self.curtailed_iters})"
        )


# ---------------------------------------------------------------------------
# Genetic placer
# ---------------------------------------------------------------------------

class GeneticPlacer:
    """
    Outer Genetic Algorithm loop over macro orientations.

    The placer is treated as a black box via *eval_fn*:

        eval_fn(benchmark_mod, net_data_mod) -> {"wl": float, "overflow": float}

    This decouples genetic.py from placer.py — no circular imports.

    Workflow per generation
    -----------------------
    1. Apply each chromosome → transformed (net_data, benchmark)
    2. Call eval_fn → (wl, overflow) → scalar fitness
    3. Rank by fitness (lower is better)
    4. Elites pass through unchanged
    5. Fill remainder via tournament selection + uniform crossover + mutation
    """

    def __init__(self, ga_section: dict) -> None:
        self.cfg = GAConfig(ga_section)
        self._rng = torch.Generator()
        self._rng.manual_seed(self.cfg.seed)

    def run(
        self,
        benchmark,
        net_data_base: dict,
        eval_fn,
    ) -> Chromosome:
        """
        Run the GA and return the best Chromosome found.

        Parameters
        ----------
        benchmark      : Benchmark object (num_hard_macros used for gene length)
        net_data_base  : net_data built with all-N orientation
        eval_fn        : callable (bench_mod, nd_mod) → metrics dict
        """
        num_hard = benchmark.num_hard_macros
        cfg      = self.cfg

        print(f"\n  ── GA: {cfg}")
        print(f"  ── {num_hard} hard macros × 8 orientations per gene")

        # ── Initialise population ────────────────────────────────────────
        # Always seed with the all-N default so we track improvement over
        # the current baseline.
        population: list[Chromosome] = [Chromosome.default(num_hard)]
        for _ in range(cfg.population_size - 1):
            population.append(Chromosome.random(num_hard, rng=self._rng))

        best_chrom = population[0]
        best_score = float("inf")
        history:   list[tuple[float, float]] = []   # (gen_best, gen_mean)

        for gen in range(cfg.n_generations):
            # ── Evaluate ─────────────────────────────────────────────────
            scores = self._evaluatePopulation(
                population, benchmark, net_data_base, eval_fn
            )

            # ── Track global best ─────────────────────────────────────────
            gen_best_idx = min(range(len(scores)), key=lambda i: scores[i])
            gen_best     = scores[gen_best_idx]
            if gen_best < best_score:
                best_score = gen_best
                best_chrom = population[gen_best_idx].clone()

            gen_mean = sum(scores) / len(scores)
            history.append((gen_best, gen_mean))
            print(
                f"  GA gen {gen:3d}/{cfg.n_generations}  "
                f"best={gen_best:.4f}  mean={gen_mean:.4f}  "
                f"global_best={best_score:.4f}"
            )

            # ── Breed next generation ─────────────────────────────────────
            population = self._nextGeneration(population, scores)

        print(
            f"  GA done — best_score={best_score:.4f}  "
            f"orientations={best_chrom}"
        )
        return best_chrom

    # ------------------------------------------------------------------ #
    # Population evaluation                                               #
    # ------------------------------------------------------------------ #

    def _evaluatePopulation(
        self,
        population:     list[Chromosome],
        benchmark,
        net_data_base:  dict,
        eval_fn,
    ) -> list[float]:
        """
        Evaluate every individual and return fitness scores (lower is better).

        Sequential baseline; parallel workers will be added here once the
        algorithm is validated.
        """
        return [
            self._evaluateOne(chrom, benchmark, net_data_base, eval_fn)
            for chrom in population
        ]

    def _evaluateOne(
        self,
        chromosome:    Chromosome,
        benchmark,
        net_data_base: dict,
        eval_fn,
    ) -> float:
        """Apply chromosome, run eval_fn, return scalar fitness."""
        net_data_mod, bench_mod = chromosome.applyToPlacement(
            net_data_base, benchmark
        )
        metrics  = eval_fn(bench_mod, net_data_mod)
        wl       = float(metrics.get("wl",       1e9))
        overflow = float(metrics.get("overflow", 1.0))
        return wl * self.cfg.wl_weight + overflow * self.cfg.overflow_weight

    # ------------------------------------------------------------------ #
    # Selection                                                           #
    # ------------------------------------------------------------------ #

    def _tournamentSelect(
        self,
        population: list[Chromosome],
        scores:     list[float],
    ) -> Chromosome:
        """k-tournament selection — returns the fittest individual in a random
        subset of size *k*.  Lower score wins."""
        k       = min(self.cfg.tournament_k, len(population))
        indices = torch.randint(
            len(population), (k,), generator=self._rng
        ).tolist()
        winner  = min(indices, key=lambda i: scores[i])
        return population[winner]

    # ------------------------------------------------------------------ #
    # Reproduction                                                        #
    # ------------------------------------------------------------------ #

    def _nextGeneration(
        self,
        population: list[Chromosome],
        scores:     list[float],
    ) -> list[Chromosome]:
        """
        Build the next generation:
          1. Elites: top *n_elite* individuals pass through unchanged.
          2. Offspring: tournament selection + uniform crossover + mutation
             fills the remainder of the population.
        """
        cfg    = self.cfg
        n      = len(population)
        ranked = sorted(range(n), key=lambda i: scores[i])

        # Elites survive unchanged
        next_pop: list[Chromosome] = [
            population[i].clone() for i in ranked[: cfg.n_elite]
        ]

        # Fill with offspring
        while len(next_pop) < n:
            pa = self._tournamentSelect(population, scores)
            pb = self._tournamentSelect(population, scores)
            c1, c2 = pa.uniform_crossover(pb, rng=self._rng)
            c1 = c1.mutate(cfg.mutation_rate, rng=self._rng)
            c2 = c2.mutate(cfg.mutation_rate, rng=self._rng)
            next_pop.append(c1)
            if len(next_pop) < n:
                next_pop.append(c2)

        return next_pop
