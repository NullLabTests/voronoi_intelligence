"""Shared utility functions for the voronoi_agi package."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


def normalize(points: NDArray, eps: float = 1e-12) -> NDArray:
    """Min-max normalise an array of points to [0, 1]."""
    lo = points.min(axis=0, keepdims=True)
    hi = points.max(axis=0, keepdims=True)
    return (points - lo) / (hi - lo + eps)


def pairwise_distances(points: NDArray) -> NDArray:
    """Compute pairwise Euclidean distances between all points."""
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(points))


def closest_seed(point: NDArray, seeds: NDArray) -> int:
    """Return index of the seed closest to *point*."""
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))


def sample_random_points_in_cell(
    vertices: NDArray,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Sample uniformly random points inside a convex Voronoi cell.

    Uses rejection sampling — efficient for 2D, slower in higher dimensions.
    """
    if rng is None:
        rng = np.random.default_rng()

    if vertices.shape[1] != 2:
        raise NotImplementedError("Only 2D cells supported for now.")

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    samples: list[NDArray] = []

    while len(samples) < n_points:
        candidate = rng.uniform(lo, hi)
        if _point_in_convex_polygon(candidate, vertices):
            samples.append(candidate)

    return np.array(samples)


def _point_in_convex_polygon(point: NDArray, polygon: NDArray) -> bool:
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if cross < 0:
            return False
    return True
