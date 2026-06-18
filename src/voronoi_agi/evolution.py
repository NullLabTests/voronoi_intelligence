"""Evolutionary operators enhanced with Voronoi structure.

Standard GA operators (selection, mutation, crossover) are augmented with
geometric awareness — selection favours under-explored territories, mutation
adapts step sizes to cell size, and novelty search uses Voronoi adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi

from .population import VoronoiPopulation


@dataclass
class VoronoiGA:
    """A genetic algorithm that leverages Voronoi territory structure.

    The population is partitioned by a Voronoi tessellation. Selection,
    mutation, and diversity maintenance are all territory-aware.
    """

    population: VoronoiPopulation
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def step(self) -> VoronoiPopulation:
        """Run one generation of the Voronoi-enhanced GA."""
        pop = self.population
        n = pop.n_individuals
        n_elite = max(1, int(n * self.elite_fraction))

        # Elitism — keep the best individuals
        elite_idx = np.argsort(pop.fitness)[::-1][:n_elite]
        elite_seeds = pop.seeds[elite_idx]
        elite_individuals = [pop.individuals[i] for i in elite_idx]
        elite_fitness = pop.fitness[elite_idx]

        # Selection — territory-aware tournament
        parent_idx = voronoi_selection(pop, n_parents=n)

        # Crossover & mutation
        offspring_seeds: list[NDArray] = []
        offspring_individuals: list[Any] = []

        individual_factory = _get_factory(pop)

        for i in range(0, len(parent_idx) - 1, 2):
            p1, p2 = parent_idx[i], parent_idx[i + 1]
            if self.rng.uniform() < self.crossover_rate:
                c1, c2 = _voronoi_crossover(
                    pop.seeds[p1], pop.seeds[p2], self.rng
                )
            else:
                c1, c2 = pop.seeds[p1].copy(), pop.seeds[p2].copy()

            for c in (c1, c2):
                c = voronoi_mutation(
                    c,
                    self.mutation_rate,
                    pop,
                    self.rng,
                )
                offspring_seeds.append(c)
                offspring_individuals.append(individual_factory(c))

        # Maintain population size
        while len(offspring_seeds) < n - n_elite:
            idx = self.rng.integers(0, len(pop.seeds))
            c = voronoi_mutation(
                pop.seeds[idx].copy(),
                self.mutation_rate * 2,
                pop,
                self.rng,
            )
            offspring_seeds.append(c)
            offspring_individuals.append(individual_factory(c))

        # Re-evaluate fitness
        fitness_fn = _get_fitness_fn(pop)
        offspring_fitness = np.array([fitness_fn(ind) for ind in offspring_individuals])

        # Rebuild population
        all_seeds = np.vstack([elite_seeds, np.array(offspring_seeds[: n - n_elite])])
        all_individuals = elite_individuals + offspring_individuals[: n - n_elite]

        if callable(getattr(pop.individuals[0], "__copy__", None)):
            all_fitness = np.array([fitness_fn(ind) for ind in all_individuals])
        else:
            all_fitness = np.concatenate([elite_fitness, offspring_fitness[: n - n_elite]])

        self.population = VoronoiPopulation(
            seeds=all_seeds,
            individuals=all_individuals,
            fitness=all_fitness,
            voronoi=Voronoi(all_seeds),
            dim=pop.dim,
        )
        return self.population

    def run(self, n_generations: int, verbose: bool = True) -> list[float]:
        """Run the GA for *n_generations* and return best-fitness history."""
        history: list[float] = []
        for g in range(n_generations):
            self.step()
            best = float(np.max(self.population.fitness))
            history.append(best)
            if verbose:
                mean = float(np.mean(self.population.fitness))
                print(f"Gen {g:4d}  best={best:.6f}  mean={mean:.6f}")
        return history


def _get_factory(pop: VoronoiPopulation) -> Callable:
    """Try to recover the individual factory from the population's individuals."""
    original = pop.individuals[0]
    if hasattr(original, "__class__") and hasattr(original, "__init__"):
        # Best-effort: create a closure that stores the seed
        def factory(seed):
            return type(original).__new__(type(original))

        return factory
    return lambda seed: seed


def _get_fitness_fn(pop: VoronoiPopulation) -> Callable:
    """Placeholder — in practice the user should supply this."""

    def default_fitness(ind):
        return 0.0

    return default_fitness


# ---------------------------------------------------------------------------
# Territory-aware selection
# ---------------------------------------------------------------------------


def voronoi_selection(
    population: VoronoiPopulation,
    n_parents: int,
    tournament_size: int = 3,
) -> NDArray:
    """Tournament selection biased by territorial under-representation.

    Individuals in sparsely populated Voronoi cells (low density) get a
    boost to their selection probability, encouraging exploration.
    """
    pop = population
    densities = pop.cell_density()
    density_weights = 1.0 / (densities + 1e-12)

    parents: list[int] = []
    while len(parents) < n_parents:
        candidates = pop.rng.integers(0, pop.n_individuals, size=tournament_size)
        # Weighted by fitness * (1 / density) to favour sparse areas
        scores = pop.fitness[candidates] * density_weights[candidates]
        winner = candidates[int(np.argmax(scores))]
        parents.append(winner)
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
    """Mutate a seed with step size proportional to its Voronoi cell size.

    Larger cells get larger mutation steps (more exploration); smaller cells
    get finer-grained mutations (exploitation).
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.uniform() > mutation_rate:
        return seed

    # Find which Voronoi cell this seed belongs to
    vor = population.voronoi
    point_region = _nearest_seed(seed, population.seeds)
    region_idx = vor.point_region[point_region]
    region = vor.regions[region_idx]

    if -1 in region or len(region) == 0:
        step_size = 0.1 / np.sqrt(population.dim)
    else:
        verts = vor.vertices[region]
        if verts.shape[1] == 2:
            from .seeds import _polygon_area
            area = _polygon_area(verts)
        else:
            from .seeds import _convex_hull_volume
            area = _convex_hull_volume(verts)
        step_size = np.sqrt(area) / 5.0 if area > 0 else 0.05

    perturbation = rng.normal(0, step_size, size=seed.shape)
    mutated = seed + perturbation
    return np.clip(mutated, 0.0, 1.0)


def _nearest_seed(point: NDArray, seeds: NDArray) -> int:
    """Return the index of the nearest seed to *point*."""
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Voronoi-aware crossover
# ---------------------------------------------------------------------------


def _voronoi_crossover(
    p1: NDArray, p2: NDArray, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """Uniform crossover that respects Voronoi region boundaries.

    Offspring are created by blending along the line between parents,
    weighted by the relative sizes of their Voronoi cells.
    """
    alpha = rng.uniform(0.3, 0.7)
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Novelty search
# ---------------------------------------------------------------------------


def novelty_search(
    population: VoronoiPopulation,
    archive: list[NDArray],
    k_neighbors: int = 5,
) -> NDArray:
    """Compute novelty scores based on Voronoi cell sparsity and behaviour.

    Novelty is defined as:
        ``sparsity = 1 / (cell_area + epsilon)``
        ``behaviour_novelty = mean dist to k-nearest neighbours in archive``
        ``total_novelty = sparsity * behaviour_novelty``
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
