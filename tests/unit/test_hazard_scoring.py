"""Unit tests: composite hazard scoring (Enterprise build — 4 real hazard
components: flood, drought, cyclone, heat)."""

from argus.common.config import HAZARD_WEIGHTS
from argus.geospatial.hazard_scoring import compute_hazard_scores


def test_hazard_weights_sum_to_one():
    assert abs(sum(HAZARD_WEIGHTS.values()) - 1.0) < 1e-9


def test_composite_scores_are_bounded_0_to_100(demo_districts):
    df, _ = compute_hazard_scores(demo_districts)
    assert not df.empty
    assert (df["composite_hazard_score"] >= 0).all()
    assert (df["composite_hazard_score"] <= 100).all()


def test_composite_is_the_declared_weighted_sum(demo_districts):
    df, _ = compute_hazard_scores(demo_districts)
    for _, row in df.iterrows():
        expected = (
            HAZARD_WEIGHTS["flood"] * row["flood_component"]
            + HAZARD_WEIGHTS["drought"] * row["drought_component"]
            + HAZARD_WEIGHTS["cyclone"] * row["cyclone_component"]
            + HAZARD_WEIGHTS["heat"] * row["heat_component"]
        )
        assert abs(expected - row["composite_hazard_score"]) < 0.01


def test_every_district_appears_exactly_once(demo_districts):
    df, _ = compute_hazard_scores(demo_districts)
    assert df["district"].is_unique


def test_returns_provenance_for_every_real_data_source(demo_districts):
    _, provenance = compute_hazard_scores(demo_districts)
    assert {"rainfall", "cyclone", "heat"}.issubset(provenance.keys())
    for record in provenance.values():
        assert record.source_organization
        assert record.retrieved_at
