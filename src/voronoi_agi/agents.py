"""Multi-agent coordination and territory assignment using Voronoi diagrams.

Each agent is assigned a Voronoi territory and is responsible for covering
or exploring that region. The framework supports dynamic re-tessellation
as agents appear, disappear, or move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi

from .population import _minimal_voronoi
from .utils import _point_in_convex_polygon as pip


def centroidal_voronoi_tessellation(
    seeds: NDArray,
    n_iterations: int = 50,
    density_func: Optional[Callable[[NDArray], float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Run Lloyd's algorithm to produce a centroidal Voronoi tessellation.

    Each iteration moves every seed to the centroid of its Voronoi cell,
    producing evenly sized territories. If *density_func* is provided,
    the centroid is weighted by the density (mass centroid).

    Parameters
    ----------
    seeds : ndarray of shape (n, d)
    n_iterations : int
    density_func : callable or None
        ``f(position) -> float`` — higher = more mass.
    rng : Generator or None

    Returns
    -------
    new_seeds : ndarray of shape (n, d)
    """
    if rng is None:
        rng = np.random.default_rng()
    dim = seeds.shape[1]

    for _ in range(n_iterations):
        vor = _minimal_voronoi(seeds)
        new_seeds = []
        for i in range(len(seeds)):
            if i < len(vor.point_region):
                region_idx = vor.point_region[i]
            else:
                region_idx = -1
            if region_idx >= 0 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
            else:
                region = []
            if -1 in region or len(region) == 0:
                new_seeds.append(seeds[i])
                continue

            verts = vor.vertices[region]
            if density_func is not None and dim == 2:
                lo = verts.min(axis=0)
                hi = verts.max(axis=0)
                candidates = rng.uniform(lo, hi, size=(200, dim))
                inside = np.array([pip(p, verts) for p in candidates])
                if inside.any():
                    weights = np.array([density_func(candidates[j]) for j in range(len(candidates)) if inside[j]])
                    centroid = np.average(candidates[inside], weights=weights, axis=0)
                else:
                    centroid = verts.mean(axis=0)
            else:
                centroid = verts.mean(axis=0)

            new_seeds.append(np.clip(centroid, 0.0, 1.0))
        seeds = np.array(new_seeds)
    return seeds


@dataclass
class AgentTerritory:
    """The Voronoi cell owned by a single agent."""

    agent_id: int
    seed: NDArray
    vertices: NDArray
    area: float
    neighbours: list[int]


@dataclass
class TerritorialAgent:
    """An agent that owns a Voronoi territory and acts within it.

    The agent's position is its seed (generator point). It can move within
    its territory, and the tessellation updates when agents re-position.
    """

    agent_id: int
    position: NDArray
    state: Any = None

    def act_within_territory(
        self,
        territory: AgentTerritory,
        policy: Callable[[NDArray, NDArray], NDArray],
    ) -> NDArray:
        """Execute a local policy that moves the agent within its territory.

        The *policy* function receives ``(agent_position, territory_vertices)``
        and returns a new position.
        """
        new_pos = policy(self.position, territory.vertices)
        self.position = np.clip(new_pos, 0.0, 1.0)
        return self.position


@dataclass
class MultiAgentCoverage:
    """A system of territorial agents covering a domain via Voronoi partitioning.

    Agents iteratively adjust their seeds to improve coverage, balance load,
    or respond to environmental changes.
    """

    agents: list[TerritorialAgent]
    dim: int = 2
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        self._update_tessellation()

    def _update_tessellation(self):
        seeds = np.array([a.position for a in self.agents])
        self.voronoi = _minimal_voronoi(seeds)
        self.territories = self._compute_territories()

    def _compute_territories(self) -> list[AgentTerritory]:
        pr = self.voronoi.point_region
        territories = []
        for i, agent in enumerate(self.agents):
            if i < len(pr) and pr[i] < len(self.voronoi.regions):
                region_idx = pr[i]
            else:
                region_idx = -1

            if region_idx >= 0 and region_idx < len(self.voronoi.regions):
                region = self.voronoi.regions[region_idx]
            else:
                region = []

            if region_idx == -1 or -1 in region or len(region) < 3:
                verts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
                area = 1.0
            else:
                verts = self.voronoi.vertices[region]
                if verts.shape[1] == 2:
                    from .seeds import _polygon_area

                    area = _polygon_area(verts)
                else:
                    from .seeds import _convex_hull_volume

                    area = _convex_hull_volume(verts)

            ridge_points = self.voronoi.ridge_points
            neighbours = []
            for rp in ridge_points:
                if i in rp:
                    neighbour = rp[0] if rp[1] == i else rp[1]
                    if neighbour != i:
                        neighbours.append(neighbour)

            territories.append(
                AgentTerritory(
                    agent_id=i,
                    seed=agent.position,
                    vertices=verts,
                    area=float(area),
                    neighbours=list(set(neighbours)),
                )
            )
        return territories

    @property
    def seeds(self) -> NDArray:
        return np.array([a.position for a in self.agents])

    def step_coverage(
        self,
        policy: Callable[[NDArray, NDArray], NDArray],
        density_field: Optional[Callable[[NDArray], float]] = None,
    ) -> list[AgentTerritory]:
        """One coverage-control step: each agent moves within its territory.

        If *density_field* is provided, agents weight their movement toward
        higher-density areas within their territory.
        """
        for i, agent in enumerate(self.agents):
            territory = self.territories[i]
            new_pos = agent.act_within_territory(territory, policy)

            if density_field is not None:
                density = density_field(new_pos)
                for _ in range(5):
                    candidate = new_pos + self.rng.normal(0, 0.01, size=self.dim)
                    if pip(candidate, territory.vertices):
                        if density_field(candidate) > density:
                            new_pos = candidate
                            density = density_field(candidate)

            agent.position = np.clip(new_pos, 0.0, 1.0)

        self._update_tessellation()
        return self.territories

    def imbalance(self) -> float:
        """Coefficient of variation of territory areas — 0 means perfectly balanced."""
        areas = np.array([t.area for t in self.territories])
        return float(np.std(areas) / (np.mean(areas) + 1e-12))

    def coverage_gap(self) -> float:
        """Fraction of the domain not covered by any bounded territory."""
        total_area = sum(t.area for t in self.territories)
        expected = 1.0
        return max(0.0, 1.0 - total_area / expected)


def territorial_assignment(
    positions: NDArray,
) -> tuple[Voronoi, list[AgentTerritory]]:
    """Given agent positions, compute the Voronoi tessellation and territories.

    This is a stateless utility — useful for one-off assignments.
    """
    vor = _minimal_voronoi(positions)
    pr = vor.point_region
    territories = []
    for i in range(len(positions)):
        if i < len(pr) and pr[i] < len(vor.regions):
            region_idx = pr[i]
        else:
            region_idx = -1

        if region_idx >= 0 and region_idx < len(vor.regions):
            region = vor.regions[region_idx]
        else:
            region = []

        if region_idx == -1 or -1 in region or len(region) < 3:
            verts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
            area = 1.0
        else:
            verts = vor.vertices[region]
            if verts.shape[1] == 2:
                from .seeds import _polygon_area

                area = _polygon_area(verts)
            else:
                from .seeds import _convex_hull_volume

                area = _convex_hull_volume(verts)

        neighbours = []
        for rp in vor.ridge_points:
            if i in rp:
                nb = rp[0] if rp[1] == i else rp[1]
                if nb != i:
                    neighbours.append(nb)

        territories.append(
            AgentTerritory(
                agent_id=i,
                seed=positions[i],
                vertices=verts,
                area=float(area),
                neighbours=list(set(neighbours)),
            )
        )
    return vor, territories
