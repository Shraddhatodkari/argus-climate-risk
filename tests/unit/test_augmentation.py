"""Unit tests: classic (dependency-free) satellite tile augmentation."""

import numpy as np

from argus.augmentation.diffusion_augment import augment_tile_classic
from argus.ingestion.copernicus import synthetic_tile


def test_classic_augmentation_produces_requested_count():
    green, nir = synthetic_tile(0.4, seed=1)
    out = augment_tile_classic(green, nir, n_augments=5, seed=2)
    assert len(out) == 5


def test_classic_augmentation_preserves_shape_and_range():
    green, nir = synthetic_tile(0.4, seed=1)
    for g, n in augment_tile_classic(green, nir, n_augments=3, seed=3):
        assert g.shape == green.shape
        assert n.shape == nir.shape
        assert g.min() >= 0.0 and g.max() <= 1.0
        assert n.min() >= 0.0 and n.max() <= 1.0


def test_classic_augmentation_is_deterministic_given_a_seed():
    green, nir = synthetic_tile(0.4, seed=1)
    out1 = augment_tile_classic(green, nir, n_augments=2, seed=7)
    out2 = augment_tile_classic(green, nir, n_augments=2, seed=7)
    for (g1, n1), (g2, n2) in zip(out1, out2):
        assert np.allclose(g1, g2)
        assert np.allclose(n1, n2)
