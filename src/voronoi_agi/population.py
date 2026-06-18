"""Population initialisation and diversity control using Voronoi partitioning.

Voronoi cells define natural "niches" in the search space. By placing one
individual per cell (or a controlled number), we guarantee population diversity
without requiring explicit fitness sharing or crowding distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi

from .seeds import SeedSampler, UniformSeedSampler


@dataclass
class VoronoiPopulation:
    """A population structured by a Voronoi tessellation of the search space.

    Each individual is associated with a Voronoi cell (territory). The
    cell structure provides natural diversity metrics and niching.
    """

    seeds: NDArray
    individuals: list[Any]
    fitness: NDArray
    voronoi: Voronoi
    dim: int

    @classmethod
    def from_sampler(
        cls,
        sampler: SeedSampler,
        individual_factory: Callable[[NDArray], Any],
        fitness_fn: Callable[[Any], float],
    ) -> "VoronoiPopulation":
        """Create a population by sampling seeds and instantiating individuals.

        Each seed becomes one individual. The *individual_factory* receives
        the seed position and returns an individual object; *fitness_fn*
        evaluates it.
        """
        seeds = sampler.sample()
        individuals = [individual_factory(s) for s in seeds]
        fitness = np.array([fitness_fn(ind) for ind in individuals])
        voronoi = Voronoi(seeds)
        return cls(
            seeds=seeds,
            individuals=individuals,
            fitness=fitness,
            voronoi=voronoi,
            dim=sampler.dim,
        )

    @property
    def n_individuals(self) -> int:
        return len(self.individuals)

    def cell_areas(self) -> NDArray:
        """Return the area/volume of each Voronoi cell."""
        from .seeds import seed_region_area

        areas = np.array([
            seed_region_area(self.voronoi, i)
            for i in range(len(self.voronoi.point_region))
        ])
        return areas

    def cell_density(self) -> NDArray:
        """Inverse of cell area — how "crowded" each region is."""
        areas = self.cell_areas()
        areas[areas == 0] = np.min(areas[areas > 0]) if np.any(areas > 0) else 1.0
        return 1.0 / areas

    def territorial_diversity(self) -> float:
        """Mean pairwise distance between seed points — a proxy for diversity."""
        from scipy.spatial.distance import pdist

        if self.n_individuals < 2:
            return 0.0
        return float(np.mean(pdist(self.seeds)))


def init_population_from_seeds(
    n_individuals: int,
    dim: int,
    individual_factory: Callable[[NDArray], Any],
    fitness_fn: Callable[[Any], float],
    strategy: str = "uniform",
    **sampler_kwargs,
) -> VoronoiPopulation:
    """Convenience function to build a Voronoi-structured population.

    Parameters
    ----------
    n_individuals : int
        Population size (number of seeds).
    dim : int
        Dimensionality of the search space.
    individual_factory : callable
        ``individual_factory(seed_position) -> individual``
    fitness_fn : callable
        ``fitness_fn(individual) -> float``
    strategy : str
        One of ``"uniform"``, ``"poisson"``, ``"gaussian"``, ``"sobol"``.
    **sampler_kwargs
        Passed to the chosen SeedSampler.
    """
    sampler_cls = {
        "uniform": UniformSeedSampler,
        "poisson": "PoissonDiskSeedSampler",
        "gaussian": "GaussianSeedSampler",
        "sobol": "SobolSeedSampler",
    }.get(strategy)

    if sampler_cls is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(sampler_cls.keys())}")

    if isinstance(sampler_cls, str):
        from .seeds import PoissonDiskSeedSampler, GaussianSeedSampler, SobolSeedSampler
        sampler_cls = eval(sampler_cls)

    sampler = sampler_cls(n_seeds=n_individuals, dim=dim, **sampler_kwargs)
    return VoronoiPopulation.from_sampler(sampler, individual_factory, fitness_fn)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


def diversity_metrics(
    population: VoronoiPopulation,
) -> dict[str, float]:
    """Compute a set of diversity metrics for a Voronoi-structured population.

    Returns
    -------
    dict with keys:
        - ``avg_pairwise_distance``
        - ``territorial_diversity``
        - ``mean_cell_area``
        - ``cell_area_cv`` (coefficient of variation)
        - ``fitness_diversity`` (fitness std / mean)
    """
    from scipy.spatial.distance import pdist

    metrics: dict[str, float] = {}
    pop = population

    if pop.n_individuals > 1:
        metrics["avg_pairwise_distance"] = float(np.mean(pdist(pop.seeds)))
    else:
        metrics["avg_pairwise_distance"] = 0.0

    metrics["territorial_diversity"] = pop.territorial_diversity()

    areas = pop.cell_areas()
    metrics["mean_cell_area"] = float(np.mean(areas))
    metrics["cell_area_cv"] = float(np.std(areas) / (np.mean(areas) + 1e-12))

    fit = pop.fitness
    metrics["fitness_diversity"] = float(np.std(fit) / (np.mean(fit) + 1e-12))

    return metrics


# ---------------------------------------------------------------------------
# Territorial niching
# ---------------------------------------------------------------------------


def territorial_niching(
    population: VoronoiPopulation,
    fitness_share_radius: Optional[float] = None,
) -> NDArray:
    """Apply fitness sharing based on Voronoi territory membership.

    Individuals in the same Voronoi cell (or neighbouring cells within
    *fitness_share_radius*) have their fitness discounted to encourage
    exploration of under-sampled regions.

    Returns a "shared fitness" array of shape ``(n_individuals,)``.
    """
    if fitness_share_radius is None:
        from scipy.spatial.distance import pdist
        if population.n_individuals > 1:
            distances = pdist(population.seeds)
            fitness_share_radius = float(np.median(distances)) * 0.5
        else:
            fitness_share_radius = 1.0

    shared = population.fitness.copy()
    for i in range(population.n_individuals):
        niche_count = 1
        for j in range(population.n_individuals):
            if i == j:
                continue
            dist = np.linalg.norm(population.seeds[i] - population.seeds[j])
            if dist < fitness_share_radius:
                niche_count += 1
        shared[i] /= niche_count
    return shared
