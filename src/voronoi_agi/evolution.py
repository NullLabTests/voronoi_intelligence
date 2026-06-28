"""Evolutionary operators enhanced with Voronoi territory structure.

The genotype is a seed (generator point) in :math:`[0,1]^d`. The Voronoi
tessellation of the population provides territory metadata (cell area,
neighbors) that adapts mutation step sizes and restricts crossover to
spatially adjacent individuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .population import VoronoiPopulation, _minimal_voronoi


# ---------------------------------------------------------------------------
# Standard GA baseline (no territory awareness)
# ---------------------------------------------------------------------------


@dataclass
class StandardGA:
    """A canonical real-valued GA with no territory awareness.

    Used as a baseline for benchmarking Voronoi-enhanced operators.
    """

    pop_size: int
    dim: int
    fitness_fn: Callable[[NDArray], float]
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    bounds: NDArray = field(default_factory=lambda: np.array([[0.0, 1.0]]))
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        lo = np.full(self.dim, self.bounds[0, 0] if self.bounds.ndim == 2 else 0.0)
        hi = np.full(self.dim, self.bounds[0, 1] if self.bounds.ndim == 2 else 1.0)
        self._lo = lo
        self._hi = hi

    def _clip(self, x: NDArray) -> NDArray:
        return np.clip(x, self._lo, self._hi)

    def _init_pop(self) -> NDArray:
        return self.rng.uniform(self._lo, self._hi, size=(self.pop_size, self.dim))

    def _evaluate(self, pop: NDArray) -> NDArray:
        return np.array([self.fitness_fn(p) for p in pop])

    def _tournament_select(self, pop: NDArray, fitness: NDArray, k: int = 3) -> NDArray:
        idx = self.rng.integers(0, self.pop_size, size=(self.pop_size, k))
        candidates = fitness[idx]
        winners = idx[np.arange(self.pop_size), candidates.argmax(axis=1)]
        return pop[winners]

    def _sbx(self, p1: NDArray, p2: NDArray, eta: float = 15.0) -> tuple[NDArray, NDArray]:
        """Simulated binary crossover (SBX)."""
        diff = np.abs(p1 - p2)
        beta = 1.0 + 2.0 * np.minimum(p1, p2) / (diff + 1e-12)
        alpha = 2.0 - beta ** (-eta - 1.0)
        u = self.rng.uniform(size=self.dim)
        beta_q = np.where(
            u <= 1.0 / alpha, (alpha * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 - alpha * u)) ** (1.0 / (eta + 1.0))
        )
        c1 = 0.5 * ((p1 + p2) - beta_q * diff)
        c2 = 0.5 * ((p1 + p2) + beta_q * diff)
        return self._clip(c1), self._clip(c2)

    def _mutate(self, x: NDArray) -> NDArray:
        """Polynomial mutation."""
        eta_m = 20.0
        r = self.rng.uniform(size=self.dim)
        delta = np.where(
            r < 0.5,
            (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0)),
        )
        return self._clip(x + delta * (self._hi - self._lo) * 0.1)

    def run(self, n_generations: int, verbose: bool = False) -> NDArray:
        """Run the GA and return fitness history (best per generation)."""
        pop = self._init_pop()
        fitness = self._evaluate(pop)
        history = np.zeros(n_generations)
        n_elite = max(1, int(self.pop_size * self.elite_fraction))

        for g in range(n_generations):
            elite_idx = np.argsort(fitness)[::-1][:n_elite]
            elites = pop[elite_idx].copy()

            parents = self._tournament_select(pop, fitness)
            offspring = []
            for i in range(0, self.pop_size - n_elite, 2):
                p1, p2 = parents[i], parents[min(i + 1, self.pop_size - 1)]
                if self.rng.uniform() < self.crossover_rate:
                    c1, c2 = self._sbx(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if self.rng.uniform() < self.mutation_rate:
                    c1 = self._mutate(c1)
                if self.rng.uniform() < self.mutation_rate:
                    c2 = self._mutate(c2)
                offspring.extend([c1, c2])

            offspring = np.array(offspring[: self.pop_size - n_elite])
            pop = np.vstack([elites, offspring])
            fitness = self._evaluate(pop)
            history[g] = float(fitness.max())
            if verbose:
                print(f"Gen {g:4d}  best={history[g]:.6f}  mean={float(fitness.mean()):.6f}")

        return history


# ---------------------------------------------------------------------------
# Fitness-sharing GA baseline
# ---------------------------------------------------------------------------


@dataclass
class FitnessSharingGA:
    """Real-valued GA with fitness sharing (Deb & Goldberg 1989).

    Niche count is computed using a distance threshold (sigma_share).
    Shared fitness = raw_fitness / niche_count.
    """

    pop_size: int
    dim: int
    fitness_fn: Callable[[NDArray], float]
    sigma_share: Optional[float] = None
    alpha_share: float = 1.0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        self._lo = np.zeros(self.dim)
        self._hi = np.ones(self.dim)

    def _clip(self, x: NDArray) -> NDArray:
        return np.clip(x, self._lo, self._hi)

    def _shared_fitness(self, pop: NDArray, raw: NDArray) -> NDArray:
        from scipy.spatial.distance import pdist, squareform

        sigma = self.sigma_share
        if sigma is None:
            dists = pdist(pop)
            sigma = float(np.median(dists)) * 0.5 if len(dists) > 0 else 0.1

        D = squareform(pdist(pop))
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.maximum(0, 1 - (D / sigma) ** self.alpha_share)
        niche = sh.sum(axis=1)
        return raw / (niche + 1e-12)

    def _init_pop(self) -> NDArray:
        return self.rng.uniform(0, 1, size=(self.pop_size, self.dim))

    def _evaluate(self, pop: NDArray) -> NDArray:
        return np.array([self.fitness_fn(p) for p in pop])

    def _tournament_select(self, fitness: NDArray, k: int = 3) -> NDArray:
        idx = self.rng.integers(0, self.pop_size, size=(self.pop_size, k))
        candidates = fitness[idx]
        winners = idx[np.arange(self.pop_size), candidates.argmax(axis=1)]
        return winners

    def _sbx(self, p1: NDArray, p2: NDArray) -> tuple[NDArray, NDArray]:
        diff = np.abs(p1 - p2)
        beta = 1.0 + 2.0 * np.minimum(p1, p2) / (diff + 1e-12)
        alpha = 2.0 - beta ** (-16.0)
        u = self.rng.uniform(size=self.dim)
        beta_q = np.where(u <= 1.0 / alpha, (alpha * u) ** (1.0 / 17.0), (1.0 / (2.0 - alpha * u)) ** (1.0 / 17.0))
        c1 = 0.5 * ((p1 + p2) - beta_q * diff)
        c2 = 0.5 * ((p1 + p2) + beta_q * diff)
        return self._clip(c1), self._clip(c2)

    def _mutate(self, x: NDArray) -> NDArray:
        r = self.rng.uniform(size=self.dim)
        delta = np.where(r < 0.5, (2.0 * r) ** (1.0 / 21.0) - 1.0, 1.0 - (2.0 * (1.0 - r)) ** (1.0 / 21.0))
        return self._clip(x + delta * 0.1)

    def run(self, n_generations: int, verbose: bool = False) -> NDArray:
        pop = self._init_pop()
        raw = self._evaluate(pop)
        fitness = self._shared_fitness(pop, raw)
        history = np.zeros(n_generations)
        n_elite = max(1, int(self.pop_size * self.elite_fraction))

        for g in range(n_generations):
            elite_idx = np.argsort(raw)[::-1][:n_elite]
            elites = pop[elite_idx].copy()

            parents_idx = self._tournament_select(fitness)
            parents = pop[parents_idx]
            offspring = []
            for i in range(0, self.pop_size - n_elite, 2):
                p1, p2 = parents[i], parents[min(i + 1, self.pop_size - 1)]
                if self.rng.uniform() < self.crossover_rate:
                    c1, c2 = self._sbx(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if self.rng.uniform() < self.mutation_rate:
                    c1 = self._mutate(c1)
                if self.rng.uniform() < self.mutation_rate:
                    c2 = self._mutate(c2)
                offspring.extend([c1, c2])

            offspring = np.array(offspring[: self.pop_size - n_elite])
            pop = np.vstack([elites, offspring])
            raw = self._evaluate(pop)
            fitness = self._shared_fitness(pop, raw)
            history[g] = float(raw.max())
            if verbose:
                print(f"Gen {g:4d}  best={history[g]:.6f}  mean={float(raw.mean()):.6f}")

        return history


# ---------------------------------------------------------------------------
# Voronoi-enhanced GA
# ---------------------------------------------------------------------------


@dataclass
class VoronoiGA:
    """Genetic algorithm with territory-aware operators.

    The genotype **is** the seed position. The Voronoi tessellation of the
    population provides:

    * **Adaptive mutation** — step size proportional to cell area
    * **Neighbor-restricted crossover** — offspring are only created between
      seeds in adjacent Voronoi cells, preserving local structure
    * **Territory-aware selection** — sparse cells get a boost
    """

    pop_size: int
    dim: int
    fitness_fn: Callable[[NDArray], float]
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    neighbor_crossover_only: bool = True
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        self._lo = np.zeros(self.dim)
        self._hi = np.ones(self.dim)

    def _init_pop(self) -> VoronoiPopulation:
        seeds = self.rng.uniform(0, 1, size=(self.pop_size, self.dim))
        individuals = [s.copy() for s in seeds]
        fitness = np.array([self.fitness_fn(s) for s in seeds])
        vor = _minimal_voronoi(seeds)
        return VoronoiPopulation(
            seeds=seeds,
            individuals=individuals,
            fitness=fitness,
            voronoi=vor,
            dim=self.dim,
            rng=self.rng,
        )

    def step(self) -> VoronoiPopulation:
        if not hasattr(self, "population"):
            self.population = self._init_pop()
        pop = self.population
        n = pop.n_individuals
        if n < 2:
            return pop
        n_elite = max(1, int(n * self.elite_fraction))

        elite_idx = np.argsort(pop.fitness)[::-1][:n_elite]
        elite_seeds = pop.seeds[elite_idx].copy()

        parent_idx = voronoi_selection(pop, n_parents=n, rng=self.rng)
        neighbors = pop.cell_neighbors()

        offspring_seeds: list[NDArray] = []
        i = 0
        while len(offspring_seeds) < n - n_elite:
            if i >= len(parent_idx):
                i = 0
            p1_idx = parent_idx[i % len(parent_idx)]
            i += 1

            if self.neighbor_crossover_only and len(neighbors[p1_idx]) > 0:
                candidate_neighbors = [nb for nb in neighbors[p1_idx] if nb < n]
                if candidate_neighbors:
                    p2_idx = self.rng.choice(candidate_neighbors)
                else:
                    p2_idx = parent_idx[(i + 1) % len(parent_idx)]
            else:
                p2_idx = parent_idx[(i + 1) % len(parent_idx)]

            p1, p2 = pop.seeds[p1_idx], pop.seeds[p2_idx]
            if self.rng.uniform() < self.crossover_rate:
                c1, c2 = _simulated_binary_crossover(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = voronoi_mutation(c1, self.mutation_rate, pop, self.rng)
            c2 = voronoi_mutation(c2, self.mutation_rate, pop, self.rng)
            offspring_seeds.extend([c1, c2])

        offspring_seeds = offspring_seeds[: n - n_elite]
        all_seeds = np.vstack([elite_seeds, np.array(offspring_seeds)])
        all_fitness = np.array([self.fitness_fn(s) for s in all_seeds])

        self.population = VoronoiPopulation(
            seeds=all_seeds,
            individuals=[s.copy() for s in all_seeds],
            fitness=all_fitness,
            voronoi=_minimal_voronoi(all_seeds),
            dim=self.dim,
            rng=self.rng,
        )
        return self.population

    def run(self, n_generations: int, verbose: bool = True) -> NDArray:
        """Run for *n_generations* and return best-fitness array."""
        self.population = self._init_pop()
        history = np.zeros(n_generations)
        for g in range(n_generations):
            self.step()
            history[g] = float(self.population.fitness.max())
            if verbose:
                mean = float(self.population.fitness.mean())
                print(f"Gen {g:4d}  best={history[g]:.6f}  mean={mean:.6f}")
        return history


# ---------------------------------------------------------------------------
# Territory-aware selection
# ---------------------------------------------------------------------------


def voronoi_selection(
    population: VoronoiPopulation,
    n_parents: int,
    tournament_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Tournament selection with a bonus for seeds in sparse territories.

    Individuals in large Voronoi cells (low density) get a boost to
    encourage exploration of under-sampled regions.
    """
    if rng is None:
        rng = np.random.default_rng()
    pop = population
    densities = pop.cell_density()
    density_weights = 1.0 / (densities + 1e-12)

    parents: list[int] = []
    while len(parents) < n_parents:
        candidates = rng.integers(0, pop.n_individuals, size=tournament_size)
        scores = pop.fitness[candidates] * density_weights[candidates]
        parents.append(candidates[int(np.argmax(scores))])
    return np.array(parents)


# ---------------------------------------------------------------------------
# Territory-adaptive mutation
# ---------------------------------------------------------------------------


def voronoi_mutation(
    seed: NDArray,
    mutation_rate: float,
    population: VoronoiPopulation,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Mutate a seed with step size proportional to its Voronoi cell area.

    Larger cells → larger steps (exploration).
    Smaller cells → finer steps (exploitation).
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.uniform() > mutation_rate:
        return seed

    vor = population.voronoi
    point_idx = _nearest_seed(seed, population.seeds)

    if point_idx < len(vor.point_region):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx] if region_idx < len(vor.regions) else []
    else:
        region = []
        region_idx = -1

    if region_idx == -1 or -1 in region or len(region) < 3:
        step_size = 0.1 / np.sqrt(population.dim)
    else:
        verts = vor.vertices[region]
        if verts.shape[1] == 2:
            from .seeds import _polygon_area

            area = _polygon_area(verts)
        else:
            from .seeds import _convex_hull_volume

            area = _convex_hull_volume(verts)
        step_size = np.sqrt(max(area, 1e-12)) / 5.0

    perturbation = rng.normal(0, step_size, size=seed.shape)
    mutated = seed + perturbation
    return np.clip(mutated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------


def _simulated_binary_crossover(
    p1: NDArray, p2: NDArray, rng: np.random.Generator, eta: float = 15.0
) -> tuple[NDArray, NDArray]:
    """Simulated binary crossover (SBX) for real-coded GAs."""
    diff = np.abs(p1 - p2)
    beta = 1.0 + 2.0 * np.minimum(p1, p2) / (diff + 1e-12)
    alpha = 2.0 - beta ** (-eta - 1.0)
    u = rng.uniform(size=p1.shape)
    beta_q = np.where(
        u <= 1.0 / alpha,
        (alpha * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 - alpha * u)) ** (1.0 / (eta + 1.0)),
    )
    c1 = 0.5 * ((p1 + p2) - beta_q * diff)
    c2 = 0.5 * ((p1 + p2) + beta_q * diff)
    return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)


def _nearest_seed(point: NDArray, seeds: NDArray) -> int:
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Novelty search (Voronoi-enhanced)
# ---------------------------------------------------------------------------


def novelty_search(
    population: VoronoiPopulation,
    archive: list[NDArray],
    k_neighbors: int = 5,
) -> NDArray:
    """Compute novelty scores: ``sparsity * behaviour_novelty``.

    *Sparsity* is the inverse of cell area. *Behaviour novelty* is the
    mean distance to the *k* nearest neighbours in *archive*.
    """
    from scipy.spatial import KDTree

    areas = population.cell_areas()
    sparsity = 1.0 / (areas + 1e-12)

    behaviour_novelty = np.zeros(population.n_individuals)
    if len(archive) > 0:
        tree = KDTree(np.array(archive))
        for i, seed in enumerate(population.seeds):
            dists, _ = tree.query(seed.reshape(1, -1), k=min(k_neighbors, len(archive)))
            behaviour_novelty[i] = float(np.mean(dists))

    return sparsity * behaviour_novelty
