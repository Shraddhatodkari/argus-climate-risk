"""Property-based edge-case tests (Hypothesis): rather than a handful of hand-picked
examples, these generate many random-but-valid satellite reflectance tiles and assert
invariants that must hold for *every* one — including extreme values a hand-written
test would likely never think to try (all-zero reflectance, all-water, all-land,
values at the numeric boundary).
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from argus.explainability.feature_contributions import explain
from argus.geospatial.flood_index import compute_ndwi, flood_extent_score

reflectance = arrays(dtype=np.float64, shape=(16, 16), elements=st.floats(0.0, 1.0, allow_nan=False))


@given(green=reflectance, nir=reflectance)
@settings(max_examples=50)
def test_ndwi_is_always_finite_and_bounded(green, nir):
    ndwi = compute_ndwi(green, nir)
    assert np.all(np.isfinite(ndwi))
    assert np.all(ndwi >= -1.01) and np.all(ndwi <= 1.01)


@given(green=reflectance, nir=reflectance)
@settings(max_examples=50)
def test_flood_extent_score_always_in_0_100(green, nir):
    score = flood_extent_score(green, nir)
    assert 0.0 <= score <= 100.0


@given(
    flood=st.floats(0.0, 100.0),
    drought=st.floats(0.0, 100.0),
    cyclone=st.floats(0.0, 100.0),
    heat=st.floats(0.0, 100.0),
)
@settings(max_examples=50)
def test_explainability_reconstructs_composite_for_any_valid_components(flood, drought, cyclone, heat):
    from argus.common.config import HAZARD_WEIGHTS

    composite = (
        HAZARD_WEIGHTS["flood"] * flood
        + HAZARD_WEIGHTS["drought"] * drought
        + HAZARD_WEIGHTS["cyclone"] * cyclone
        + HAZARD_WEIGHTS["heat"] * heat
    )
    row = {
        "flood_component": flood,
        "drought_component": drought,
        "cyclone_component": cyclone,
        "heat_component": heat,
        "composite_hazard_score": composite,
    }
    contributions = explain(row)
    assert abs(sum(c.contribution for c in contributions) - composite) < 0.05
