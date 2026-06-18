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

from .seeds import (
    SeedSampler,
    UniformSeedSampler,
    PoissonDiskSeedSampler,
    GaussianSeedSampler,
    SobolSeedSampler,
)


def _minimal_voronoi(seeds: NDArray) -> Voronoi:
    """Build a Voronoi tessellation, handling edge cases with too few points.

    Qhull requires at least ``dim + 1`` non-degenerate points. For fewer
    points we add synthetic jittered seeds to satisfy Qhull, then trim
    the result.
    """
    n, d = seeds.shape
    if n >= d + 1:
        try:
            return Voronoi(seeds)
        except Exception:
            pass
    needed = d + 1 - n
    if needed > 0:
        extra = np.eye(d, dtype=float) * 0.05 + 0.5
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.01, 0.01, size=(d, d))
        extra = np.clip(extra + jitter, 0, 1)
        extra = extra[:needed]
        combined = np.vstack([seeds, extra])
        vor = Voronoi(combined)
        return vor
    return Voronoi(seeds)


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
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

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
        voronoi = _minimal_voronoi(seeds)
        return cls(
            seeds=seeds,
            individuals=individuals,
            fitness=fitness,
            voronoi=voronoi,
            dim=sampler.dim,
            rng=sampler.rng,
        )

    @property
    def n_individuals(self) -> int:
        return len(self.individuals)

    def cell_areas(self) -> NDArray:
        """Return the area/volume of each Voronoi cell."""
        from .seeds import seed_region_area

        pr = self.voronoi.point_region
        areas = np.array([
            seed_region_area(self.voronoi, pr[i])
            for i in range(len(pr))
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

    def cell_neighbors(self) -> list[list[int]]:
        """Return adjacency list: ``neighbors[i]`` lists indices of cells adjacent to cell *i*.

        Two cells are neighbours if they share a Voronoi ridge (edge in 2D).
        """
        n = len(self.seeds)
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for (i, j) in self.voronoi.ridge_points:
            if i < n and j < n:
                neighbors[i].append(int(j))
                neighbors[j].append(int(i))
        return [list(set(nb)) for nb in neighbors]

    def cell_vertices(self) -> list[NDArray]:
        """Return the vertex array for each bounded Voronoi cell.

        Unbounded cells return an empty array.
        """
        verts_list: list[NDArray] = []
        for i in range(len(self.seeds)):
            region_idx = self.voronoi.point_region[i]
            if region_idx < len(self.voronoi.regions):
                region = self.voronoi.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    verts_list.append(self.voronoi.vertices[region])
                    continue
            verts_list.append(np.array([]))
        return verts_list


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
    registry = {
        "uniform": UniformSeedSampler,
        "poisson": PoissonDiskSeedSampler,
        "gaussian": GaussianSeedSampler,
        "sobol": SobolSeedSampler,
    }
    sampler_cls = registry.get(strategy)
    if sampler_cls is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(registry)}")

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
