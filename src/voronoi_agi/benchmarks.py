"""Benchmark comparing VoronoiGA vs StandardGA vs FitnessSharingGA.

Test functions (all adapted to [0,1]^d, maximised):
  - Sphere (simple, unimodal)
  - Rastrigin (many local minima)
  - Ackley (steep outer region, flat centre)
  - Rosenbrock (narrow valley)

Usage:
    from voronoi_agi.benchmarks import run_comparison
    results = run_comparison(dim=2, pop_size=50, n_gens=100)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .evolution import StandardGA, FitnessSharingGA, VoronoiGA


# ---------------------------------------------------------------------------
# Test functions (maximise → higher is better)
# ---------------------------------------------------------------------------


def sphere(x: NDArray) -> float:
    """Sphere: simple unimodal, optimum at (0.5, ..., 0.5)."""
    return -float(np.sum((x - 0.5) ** 2))


def rastrigin(x: NDArray) -> float:
    """Rastrigin: many local minima, optimum at (0.5, ..., 0.5)."""
    scaled = x * 10 - 5
    d = len(scaled)
    A = 10.0
    return -(A * d + np.sum(scaled ** 2 - A * np.cos(2 * np.pi * scaled)))


def ackley(x: NDArray) -> float:
    """Ackley: steep outer, flat centre, optimum at (0.5, ..., 0.5)."""
    shifted = x - 0.5
    d = len(shifted)
    sum_sq = np.sum(shifted ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * shifted))
    val = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20.0 + np.e
    return -float(val)


def rosenbrock(x: NDArray) -> float:
    """Rosenbrock valley: narrow ridge, optimum at (0.5, ..., 0.5)."""
    shifted = x * 10 - 5 + 1  # shift so optimum of original is at x=1 → x=0.5 in [0,1]
    d = len(shifted)
    val = sum(
        100.0 * (shifted[i + 1] - shifted[i] ** 2) ** 2 + (1.0 - shifted[i]) ** 2
        for i in range(d - 1)
    )
    return -float(val)


FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Results from a single algorithm run on one test function."""

    algorithm: str
    function: str
    dim: int
    pop_size: int
    n_generations: int
    fitness_history: NDArray
    best_fitness: float
    final_diversity: float = 0.0


def run_single(
    algorithm: str,
    function_name: str,
    dim: int = 2,
    pop_size: int = 50,
    n_generations: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    """Run one algorithm on one test function and return results."""
    fn = FUNCTIONS[function_name]
    rng = np.random.default_rng(seed)

    if algorithm == "standard":
        ga = StandardGA(pop_size=pop_size, dim=dim, fitness_fn=fn, rng=rng)
    elif algorithm == "sharing":
        ga = FitnessSharingGA(pop_size=pop_size, dim=dim, fitness_fn=fn, rng=rng)
    elif algorithm == "voronoi":
        ga = VoronoiGA(pop_size=pop_size, dim=dim, fitness_fn=fn, rng=rng)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    history = ga.run(n_generations=n_generations, verbose=False)

    if hasattr(ga, "population"):
        pop = ga.population
        from scipy.spatial.distance import pdist
        diversity = float(np.mean(pdist(pop.seeds))) if pop.n_individuals > 1 else 0.0
    else:
        diversity = 0.0

    return BenchmarkResult(
        algorithm=algorithm,
        function=function_name,
        dim=dim,
        pop_size=pop_size,
        n_generations=n_generations,
        fitness_history=history,
        best_fitness=float(history[-1]),
        final_diversity=diversity,
    )


def run_comparison(
    dim: int = 2,
    pop_size: int = 50,
    n_generations: int = 100,
    n_trials: int = 5,
    algorithms: tuple[str, ...] = ("standard", "sharing", "voronoi"),
    functions: tuple[str, ...] = ("sphere", "rastrigin", "ackley", "rosenbrock"),
) -> list[list[BenchmarkResult]]:
    """Run all algorithms on all functions and return results table.

    Returns a list (per function) of lists (per algorithm) of BenchmarkResult.
    """
    all_results: list[list[BenchmarkResult]] = []
    for fn_name in functions:
        fn_results: list[BenchmarkResult] = []
        for algo in algorithms:
            trial_results = []
            for trial in range(n_trials):
                result = run_single(
                    algorithm=algo,
                    function_name=fn_name,
                    dim=dim,
                    pop_size=pop_size,
                    n_generations=n_generations,
                    seed=42 + trial,
                )
                trial_results.append(result)
            fn_results.append(trial_results[0])
        all_results.append(fn_results[0])
    return all_results


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Pretty-print a comparison table."""
    print(f"{'Function':<15} {'Algorithm':<12} {'Best':<12} {'Gens':<6}")
    print("-" * 55)
    for r in results:
        print(f"{r.function:<15} {r.algorithm:<12} {r.best_fitness:<12.6f} {r.n_generations:<6}")
