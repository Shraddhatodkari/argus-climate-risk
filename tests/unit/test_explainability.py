"""Unit tests: hazard-score explainability decomposition (4 hazards: flood,
drought, cyclone, heat)."""

import pytest

from argus.explainability.feature_contributions import explain
from argus.geospatial.exposure_engine import compute_exposure_at_risk


def test_contributions_reconstruct_the_composite_score(demo_districts):
    df = compute_exposure_at_risk(demo_districts)
    for _, row in df.iterrows():
        contributions = explain(row.to_dict())
        total = sum(c.contribution for c in contributions)
        assert abs(total - row["composite_hazard_score"]) < 0.05


def test_contributions_sorted_descending(demo_districts):
    df = compute_exposure_at_risk(demo_districts)
    row = df.iloc[0].to_dict()
    contributions = explain(row)
    values = [c.contribution for c in contributions]
    assert values == sorted(values, reverse=True)


def test_all_four_hazards_present(demo_districts):
    df = compute_exposure_at_risk(demo_districts)
    row = df.iloc[0].to_dict()
    hazards = {c.hazard for c in explain(row)}
    assert hazards == {"flood", "drought", "cyclone", "heat"}


def test_mismatched_composite_raises():
    bad_row = {
        "flood_component": 50.0,
        "drought_component": 50.0,
        "cyclone_component": 50.0,
        "heat_component": 50.0,
        "composite_hazard_score": 999.0,  # deliberately inconsistent
    }
    with pytest.raises(AssertionError):
        explain(bad_row)
