"""Tests for the seeds module."""

import numpy as np
from voronoi_agi.seeds import (
    UniformSeedSampler,
    PoissonDiskSeedSampler,
    GaussianSeedSampler,
    SobolSeedSampler,
)


class TestSeedSamplers:
    def test_uniform_shape(self):
        sampler = UniformSeedSampler(n_seeds=100, dim=2)
        seeds = sampler.sample()
        assert seeds.shape == (100, 2)

    def test_uniform_bounds(self):
        sampler = UniformSeedSampler(n_seeds=50, dim=2, bounds=np.array([[0, 1], [0, 1]]))
        seeds = sampler.sample()
        assert seeds.min() >= 0 and seeds.max() <= 1

    def test_uniform_custom_bounds(self):
        sampler = UniformSeedSampler(
            n_seeds=50,
            dim=2,
            bounds=np.array([[-1, 1], [-2, 2]]),
        )
        seeds = sampler.sample()
        assert seeds[:, 0].min() >= -1 and seeds[:, 0].max() <= 1
        assert seeds[:, 1].min() >= -2 and seeds[:, 1].max() <= 2

    def test_poisson_disk(self):
        sampler = PoissonDiskSeedSampler(n_seeds=30, dim=2, radius=0.15)
        seeds = sampler.sample()
        assert seeds.shape == (30, 2)

    def test_gaussian(self):
        sampler = GaussianSeedSampler(n_seeds=50, dim=2, n_centers=3)
        seeds = sampler.sample()
        assert seeds.shape == (50, 2)

    def test_sobol(self):
        sampler = SobolSeedSampler(n_seeds=64, dim=3)
        seeds = sampler.sample()
        assert seeds.shape == (64, 3)

    def test_high_dim(self):
        sampler = UniformSeedSampler(n_seeds=20, dim=10)
        seeds = sampler.sample()
        assert seeds.shape == (20, 10)

    def test_reproducible(self):
        sampler1 = UniformSeedSampler(n_seeds=50, dim=2, rng=np.random.default_rng(42))
        sampler2 = UniformSeedSampler(n_seeds=50, dim=2, rng=np.random.default_rng(42))
        assert np.allclose(sampler1.sample(), sampler2.sample())

    def test_bounds_validation(self):
        try:
            UniformSeedSampler(n_seeds=5, dim=2, bounds=np.array([[0, 1]]))
            assert False, "Should have raised"
        except ValueError:
            pass
