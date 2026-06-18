"""Seed (generator point) generation and management for Voronoi-based AGI systems.

This module provides multiple strategies for sampling generator points ("seeds")
that drive Voronoi tessellations. Seeds are the fundamental primitive — they
define regions, territories, and population structure in the search space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi
from scipy.stats import qmc


@dataclass
class SeedSampler(ABC):
    """Abstract base for all seed sampling strategies.

    A seed sampler produces an array of generator points (seeds) in
    :math:`[0, 1]^d` that can be used to construct a Voronoi tessellation.
    """

    n_seeds: int
    dim: int = 2
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    bounds: Optional[NDArray] = None

    def __post_init__(self):
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds, dtype=float)
            if self.bounds.shape != (self.dim, 2):
                raise ValueError(
                    f"bounds must have shape ({self.dim}, 2), got {self.bounds.shape}"
                )

    @abstractmethod
    def sample(self) -> NDArray:
        """Return an array of shape (n_seeds, dim) of seed positions.

        Seeds should lie in :math:`[0, 1]^d` (or within *bounds* if set).
        """
        ...

    def _rescale(self, points: NDArray) -> NDArray:
        if self.bounds is not None:
            lo = self.bounds[:, 0]
            hi = self.bounds[:, 1]
            points = lo + points * (hi - lo)
        return points


class UniformSeedSampler(SeedSampler):
    """Uniformly random seed positions — the simplest baseline."""

    def sample(self) -> NDArray:
        points = self.rng.uniform(0.0, 1.0, size=(self.n_seeds, self.dim))
        return self._rescale(points)


class PoissonDiskSeedSampler(SeedSampler):
    """Poisson-disk (blue noise) sampling for maximal spacing between seeds.

    Uses a simple "dart throwing" rejection strategy. Seeds are guaranteed to
    be at least *radius* apart, producing well-spaced Voronoi cells.
    """

    radius: float = 0.05
    max_attempts: int = 1000

    def sample(self) -> NDArray:
        points: list[NDArray] = []
        attempts = 0
        while len(points) < self.n_seeds and attempts < self.max_attempts:
            candidate = self.rng.uniform(0.0, 1.0, size=self.dim)
            if all(
                np.linalg.norm(candidate - p) >= self.radius for p in points
            ):
                points.append(candidate)
            attempts += 1
        if len(points) < self.n_seeds:
            remaining = self.n_seeds - len(points)
            extra = self.rng.uniform(0.0, 1.0, size=(remaining, self.dim))
            points.extend(extra)
        return self._rescale(np.array(points))


class GaussianSeedSampler(SeedSampler):
    """Seeds sampled from a Gaussian mixture, producing variable-density regions.

    Useful when you want higher seed density (finer Voronoi cells) in
    promising areas of the search space.
    """

    n_centers: int = 3
    cluster_std: float = 0.15

    def sample(self) -> NDArray:
        centers = self.rng.uniform(0.0, 1.0, size=(self.n_centers, self.dim))
        assignments = self.rng.integers(0, self.n_centers, size=self.n_seeds)
        points = np.zeros((self.n_seeds, self.dim))
        for i in range(self.n_seeds):
            pt = centers[assignments[i]] + self.rng.normal(
                0, self.cluster_std, size=self.dim
            )
            points[i] = np.clip(pt, 0.0, 1.0)
        return self._rescale(points)


class SobolSeedSampler(SeedSampler):
    """Quasi-random Sobol sequence — low-discrepancy, deterministic coverage.

    Sobol sequences provide excellent space-filling properties with minimal
    clustering, ideal for initialising populations in high dimensions.
    """

    def sample(self) -> NDArray:
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=int(self.rng.integers(0, 2**31)))
        points = sampler.random(n=self.n_seeds)
        return self._rescale(np.asarray(points))


# ---------------------------------------------------------------------------
# Seed utility functions
# ---------------------------------------------------------------------------


def seed_region_area(vor: Voronoi, region_idx: int) -> float:
    """Return the area (2D) or volume (nD) of a single Voronoi region."""
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        return 0.0
    vertices = vor.vertices[region]
    if vertices.shape[1] == 2:
        return _polygon_area(vertices)
    return _convex_hull_volume(vertices)


def seed_region_vertices(vor: Voronoi, region_idx: int) -> Optional[NDArray]:
    """Return the vertices of a single Voronoi region, or None if unbounded."""
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        return None
    return vor.vertices[region]


def _polygon_area(vertices: NDArray) -> float:
    """Shoelace formula for 2D polygon area."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _convex_hull_volume(vertices: NDArray) -> float:
    """Approximate volume via convex hull (scipy)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    return hull.volume
