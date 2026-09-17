"""Unit tests: Climate-Adjusted Exposure at Risk (Enterprise build)."""

from argus.common.config import HIGH_RISK_THRESHOLD
from argus.geospatial.exposure_engine import compute_exposure_at_risk, run_pipeline


def test_climate_exposed_exposure_never_exceeds_public_exposure(demo_districts):
    df = compute_exposure_at_risk(demo_districts)
    assert (df["climate_exposed_exposure_inr_cr"] <= df["public_exposure_inr_cr"] + 0.01).all()


def test_high_risk_flag_matches_threshold(demo_districts):
    df = compute_exposure_at_risk(demo_districts)
    for _, row in df.iterrows():
        assert row["is_high_risk"] == (row["composite_hazard_score"] >= HIGH_RISK_THRESHOLD)


def test_at_least_one_high_risk_district_in_the_demo_sample(demo_districts):
    # The demo dataset deliberately includes real flood/cyclone-prone districts
    # (Kerala, Odisha, Tamil Nadu coast) — the pipeline should surface at least one.
    df = compute_exposure_at_risk(demo_districts)
    assert df["is_high_risk"].any()


def test_sorted_descending_by_exposure(demo_districts):
    df = compute_exposure_at_risk(demo_districts)
    values = df["climate_exposed_exposure_inr_cr"].tolist()
    assert values == sorted(values, reverse=True)


def test_run_pipeline_returns_every_intermediate_table(demo_districts):
    output = run_pipeline(demo_districts)
    assert not output.hazard_scores.empty
    assert not output.public_exposure.empty
    assert not output.sector_exposure.empty
    assert not output.impact_detail.empty
    assert not output.impact_by_district.empty
    assert not output.exposure_at_risk.empty
    assert set(output.exposure_at_risk["district"]) == set(demo_districts["district"])
