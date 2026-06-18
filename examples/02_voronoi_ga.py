#!/usr/bin/env python3
"""Example 2: Voronoi-enhanced genetic algorithm on continuous benchmarks.

Solves the Rastrigin function (shifted to [0,1]^d) using territory-aware
selection, mutation, and niching.
"""

import numpy as np
from scipy.spatial import Voronoi

from voronoi_agi.population import VoronoiPopulation
from voronoi_agi.evolution import VoronoiGA
from voronoi_agi.seeds import UniformSeedSampler

# Rastrigin function adapted to [0, 1]^d domain
# minimum at f(0.5) = 0, harder with many local minima
def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    scaled = x * 10 - 5  # map [0,1] -> [-5, 5]
    d = len(scaled)
    return -(
        A * d
        + np.sum(scaled ** 2 - A * np.cos(2 * np.pi * scaled))
    )  # negative = maximise


DIM = 5
N_POP = 80
N_GENS = 100

sampler = UniformSeedSampler(n_seeds=N_POP, dim=DIM, rng=np.random.default_rng(42))
pop = VoronoiPopulation.from_sampler(
    sampler,
    individual_factory=lambda s: s,
    fitness_fn=rastrigin,
)

ga = VoronoiGA(population=pop, mutation_rate=0.2, rng=np.random.default_rng(42))
history = ga.run(n_generations=N_GENS)

print(f"\nBest fitness: {history[-1]:.4f}")
print(f"Improvement: {history[-1] - history[0]:.4f}")
print(f"Final best seed: {ga.population.seeds[np.argmax(ga.population.fitness)]}")
