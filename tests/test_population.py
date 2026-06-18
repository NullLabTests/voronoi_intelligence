"""Tests for the population module."""

import numpy as np
from voronoi_agi.population import VoronoiPopulation, diversity_metrics, territorial_niching
from voronoi_agi.seeds import UniformSeedSampler


def dummy_factory(seed):
    return seed.copy()


def dummy_fitness(ind):
    return -float(np.sum((ind - 0.5) ** 2))


def make_test_population(n=20, dim=2):
    sampler = UniformSeedSampler(n_seeds=n, dim=dim, rng=np.random.default_rng(42))
    return VoronoiPopulation.from_sampler(sampler, dummy_factory, dummy_fitness)


class TestVoronoiPopulation:
    def test_from_sampler(self):
        pop = make_test_population(20)
        assert pop.n_individuals == 20
        assert pop.seeds.shape == (20, 2)

    def test_cell_areas(self):
        pop = make_test_population(20)
        areas = pop.cell_areas()
        assert areas.shape == (20,)
        assert np.all(areas >= 0)

    def test_cell_density(self):
        pop = make_test_population(20)
        density = pop.cell_density()
        assert density.shape == (20,)
        assert np.all(density > 0)

    def test_territorial_diversity(self):
        pop = make_test_population(20)
        div = pop.territorial_diversity()
        assert div > 0

    def test_single_individual_diversity(self):
        sampler = UniformSeedSampler(n_seeds=1, dim=2, rng=np.random.default_rng(42))
        pop = VoronoiPopulation.from_sampler(sampler, dummy_factory, dummy_fitness)
        assert pop.territorial_diversity() == 0.0

    def test_two_individuals(self):
        sampler = UniformSeedSampler(n_seeds=2, dim=2, rng=np.random.default_rng(42))
        pop = VoronoiPopulation.from_sampler(sampler, dummy_factory, dummy_fitness)
        assert pop.n_individuals == 2
        assert pop.territorial_diversity() > 0

    def test_cell_neighbors(self):
        pop = make_test_population(20)
        neighbors = pop.cell_neighbors()
        assert len(neighbors) == 20
        for nb in neighbors:
            assert all(isinstance(n, int) for n in nb)
            assert all(0 <= n < 20 for n in nb)

    def test_cell_vertices(self):
        pop = make_test_population(20)
        verts = pop.cell_vertices()
        assert len(verts) == 20
        assert any(len(v) > 0 for v in verts)


class TestDiversityMetrics:
    def test_metrics_return_dict(self):
        pop = make_test_population(20)
        metrics = diversity_metrics(pop)
        assert "avg_pairwise_distance" in metrics
        assert "territorial_diversity" in metrics
        assert "mean_cell_area" in metrics
        assert metrics["avg_pairwise_distance"] > 0

    def test_single_individual(self):
        sampler = UniformSeedSampler(n_seeds=1, dim=2, rng=np.random.default_rng(42))
        pop = VoronoiPopulation.from_sampler(sampler, dummy_factory, dummy_fitness)
        metrics = diversity_metrics(pop)
        assert metrics["avg_pairwise_distance"] == 0.0


class TestTerritorialNiching:
    def test_shared_fitness_shape(self):
        pop = make_test_population(20)
        shared = territorial_niching(pop)
        assert shared.shape == (20,)

    def test_shared_fitness_differs(self):
        pop = make_test_population(20)
        shared = territorial_niching(pop)
        assert not np.allclose(shared, pop.fitness)
