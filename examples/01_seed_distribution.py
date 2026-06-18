#!/usr/bin/env python3
"""Example 1: Compare different seed sampling strategies visually."""

import matplotlib.pyplot as plt
import numpy as np

from voronoi_agi.seeds import (
    UniformSeedSampler,
    PoissonDiskSeedSampler,
    GaussianSeedSampler,
    SobolSeedSampler,
)
from voronoi_agi.visualization import plot_voronoi_2d

N_SEEDS = 40
DIM = 2

samplers = {
    "Uniform": UniformSeedSampler(n_seeds=N_SEEDS, dim=DIM),
    "Poisson Disk": PoissonDiskSeedSampler(n_seeds=N_SEEDS, dim=DIM, radius=0.08),
    "Gaussian Mixture": GaussianSeedSampler(n_seeds=N_SEEDS, dim=DIM, n_centers=3, cluster_std=0.12),
    "Sobol (QMC)": SobolSeedSampler(n_seeds=N_SEEDS, dim=DIM),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, (name, sampler) in zip(axes.flat, samplers.items()):
    seeds = sampler.sample()
    plot_voronoi_2d(seeds, ax=ax, title=name, alpha=0.2)

plt.tight_layout()
plt.savefig("docs/seed_comparison.png", dpi=150, bbox_inches="tight")
print("Saved seed comparison to docs/seed_comparison.png")
plt.show()
