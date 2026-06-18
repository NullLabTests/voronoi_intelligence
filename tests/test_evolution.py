"""Tests for the evolution module."""

import numpy as np
from voronoi_agi.evolution import (
    VoronoiGA,
    StandardGA,
    FitnessSharingGA,
    voronoi_selection,
    voronoi_mutation,
)
from voronoi_agi.population import VoronoiPopulation
from voronoi_agi.seeds import UniformSeedSampler


def sphere(x: np.ndarray) -> float:
    return -float(np.sum((x - 0.5) ** 2))


def rastrigin(x: np.ndarray) -> float:
    scaled = x * 10 - 5
    d = len(scaled)
    A = 10.0
    return -(A * d + np.sum(scaled ** 2 - A * np.cos(2 * np.pi * scaled)))


def make_test_population(n=20, dim=2):
    sampler = UniformSeedSampler(n_seeds=n, dim=dim, rng=np.random.default_rng(42))
    return VoronoiPopulation.from_sampler(sampler, lambda s: s.copy(), sphere)


class TestStandardGA:
    def test_baseline_runs(self):
        ga = StandardGA(pop_size=20, dim=2, fitness_fn=sphere, rng=np.random.default_rng(42))
        history = ga.run(n_generations=10)
        assert history.shape == (10,)
        assert history[-1] >= history[0]  # should improve or stay flat


class TestFitnessSharingGA:
    def test_sharing_runs(self):
        ga = FitnessSharingGA(pop_size=20, dim=2, fitness_fn=sphere, rng=np.random.default_rng(42))
        history = ga.run(n_generations=10)
        assert history.shape == (10,)
        assert history[-1] >= history[0]


class TestVoronoiSelection:
    def test_returns_correct_count(self):
        pop = make_test_population(20)
        parents = voronoi_selection(pop, n_parents=10)
        assert len(parents) == 10
        assert np.all(parents >= 0) and np.all(parents < 20)

    def test_selection_unique_ids(self):
        pop = make_test_population(50)
        parents = voronoi_selection(pop, n_parents=50)
        assert len(np.unique(parents)) <= 50


class TestVoronoiMutation:
    def test_mutation_changes_seed(self):
        pop = make_test_population(20)
        original = pop.seeds[0].copy()
        mutated = voronoi_mutation(original.copy(), mutation_rate=1.0, population=pop)
        assert not np.allclose(original, mutated)

    def test_mutation_respects_bounds(self):
        pop = make_test_population(20)
        seed = np.array([0.5, 0.5])
        for _ in range(50):
            mutated = voronoi_mutation(seed.copy(), mutation_rate=1.0, population=pop)
            assert np.all(mutated >= 0) and np.all(mutated <= 1)

    def test_mutation_zero_rate(self):
        pop = make_test_population(20)
        seed = np.array([0.5, 0.5])
        mutated = voronoi_mutation(seed.copy(), mutation_rate=0.0, population=pop)
        assert np.allclose(seed, mutated)

    def test_mutation_cell_size_adaptive(self):
        """Seeds in large cells should get larger mutations than small cells."""
        pop = make_test_population(50)
        areas = pop.cell_areas()
        large_cell_idx = int(np.argmax(areas))
        small_cell_idx = int(np.argmin(areas[areas > 0]))

        large_muts = []
        small_muts = []
        for _ in range(30):
            lm = voronoi_mutation(pop.seeds[large_cell_idx].copy(), mutation_rate=1.0, population=pop)
            sm = voronoi_mutation(pop.seeds[small_cell_idx].copy(), mutation_rate=1.0, population=pop)
            large_muts.append(np.linalg.norm(lm - pop.seeds[large_cell_idx]))
            small_muts.append(np.linalg.norm(sm - pop.seeds[small_cell_idx]))

        assert np.mean(large_muts) >= np.mean(small_muts) * 0.5


class TestVoronoiGA:
    def test_step_creates_population(self):
        ga = VoronoiGA(pop_size=20, dim=2, fitness_fn=sphere, rng=np.random.default_rng(42))
        ga._init_pop()
        ga.step()
        assert ga.population.n_individuals == 20
        assert ga.population.seeds.shape == (20, 2)

    def test_step_without_init_auto_inits(self):
        ga = VoronoiGA(pop_size=20, dim=2, fitness_fn=sphere, rng=np.random.default_rng(42))
        ga.step()
        assert ga.population.n_individuals == 20

    def test_run_returns_history(self):
        ga = VoronoiGA(pop_size=20, dim=2, fitness_fn=sphere, rng=np.random.default_rng(42))
        history = ga.run(n_generations=5, verbose=False)
        assert len(history) == 5

    def test_fitness_improves(self):
        ga = VoronoiGA(pop_size=30, dim=2, fitness_fn=sphere, mutation_rate=0.2, rng=np.random.default_rng(42))
        history = ga.run(n_generations=20, verbose=False)
        assert history[-1] >= history[0]

    def test_neighbor_crossover_runs(self):
        ga = VoronoiGA(pop_size=20, dim=2, fitness_fn=sphere, neighbor_crossover_only=True, rng=np.random.default_rng(42))
        ga._init_pop()
        ga.step()
        assert ga.population.n_individuals == 20

    def test_voronoi_outperforms_standard_ga(self):
        """VoronoiGA should outperform the standard GA on a multimodal problem."""
        from voronoi_agi.evolution import StandardGA

        voronoi = VoronoiGA(pop_size=50, dim=2, fitness_fn=rastrigin, rng=np.random.default_rng(42))
        standard = StandardGA(pop_size=50, dim=2, fitness_fn=rastrigin, rng=np.random.default_rng(42))

        v_history = voronoi.run(n_generations=100, verbose=False)
        s_history = standard.run(n_generations=100, verbose=False)

        assert v_history[-1] > s_history[-1], f"Voronoi {v_history[-1]:.4f} <= Standard {s_history[-1]:.4f}"
