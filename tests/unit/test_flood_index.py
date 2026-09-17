"""Unit tests: NDWI computation and satellite flood-extent detection."""

import numpy as np

from argus.geospatial.flood_index import compute_ndwi, flood_extent_fraction, flood_extent_score
from argus.ingestion.copernicus import synthetic_tile


def test_ndwi_positive_for_water_negative_for_land():
    # Pure water pixel: high green, low NIR -> NDWI clearly positive.
    water_ndwi = compute_ndwi(np.array([0.30]), np.array([0.05]))
    assert water_ndwi[0] > 0.5

    # Pure land/vegetation pixel: low green, high NIR -> NDWI clearly negative.
    land_ndwi = compute_ndwi(np.array([0.12]), np.array([0.35]))
    assert land_ndwi[0] < -0.3


def test_ndwi_handles_zero_reflectance_without_error():
    ndwi = compute_ndwi(np.array([0.0]), np.array([0.0]))
    assert np.isfinite(ndwi[0])


def test_flood_extent_fraction_recovers_known_water_fraction():
    for true_fraction in (0.0, 0.2, 0.5, 0.8, 1.0):
        green, nir = synthetic_tile(true_fraction, size=128, seed=42)
        recovered = flood_extent_fraction(green, nir)
        assert abs(recovered - true_fraction) < 0.08, (
            f"expected ~{true_fraction}, got {recovered} — NDWI thresholding drifted "
            "too far from the synthetic tile's ground truth"
        )


def test_flood_extent_score_is_0_to_100_scaled():
    green, nir = synthetic_tile(0.5, seed=1)
    score = flood_extent_score(green, nir)
    assert 0.0 <= score <= 100.0
