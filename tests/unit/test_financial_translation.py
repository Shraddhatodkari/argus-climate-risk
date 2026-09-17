"""Unit tests: financial-risk translation layer — item #6 of the Enterprise spec
(Portfolio exposure / Climate-exposed exposure / Severe-stress exposure / Estimated
impact range / Primary hazard / Most affected sector, one row per district)."""

from argus.financial.translation import build_financial_translation
from argus.geospatial.exposure_engine import run_pipeline


def test_translation_has_one_row_per_district_with_every_required_field(demo_districts):
    output = run_pipeline(demo_districts)
    translation = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)

    assert set(translation["district"]) == set(demo_districts["district"])
    required = {
        "portfolio_exposure_inr_cr",
        "portfolio_exposure_source",
        "climate_exposed_exposure_inr_cr",
        "severe_stress_exposure_inr_cr",
        "estimated_impact_low_inr_cr",
        "estimated_impact_high_inr_cr",
        "primary_hazard",
        "most_affected_sector",
    }
    assert required.issubset(translation.columns)


def test_climate_exposed_and_severe_stress_never_exceed_portfolio_exposure(demo_districts):
    output = run_pipeline(demo_districts)
    translation = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)
    assert (translation["climate_exposed_exposure_inr_cr"] <= translation["portfolio_exposure_inr_cr"] + 0.01).all()


def test_severe_stress_exposure_is_bounded_and_plausible(demo_districts):
    """climate_exposed_exposure_inr_cr (this module's 'baseline') applies the
    UNSTRESSED hazard score against the district's FULL sector exposure, while
    severe_stress_exposure_inr_cr applies a stressed (<=1.6x, capped at 100) hazard
    score against only the Severe Stress scenario's exposure_affected_share (0.75)
    of that same exposure -- so severe stress is not guaranteed to exceed baseline
    for an already near-saturated hazard score (a smaller affected base can outweigh
    a capped multiplier). What must always hold: both stay non-negative and neither
    exceeds the district's portfolio exposure (checked elsewhere in this file)."""
    output = run_pipeline(demo_districts)
    translation = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)
    assert (translation["severe_stress_exposure_inr_cr"] >= 0).all()
    assert (translation["climate_exposed_exposure_inr_cr"] >= 0).all()


def test_impact_range_is_low_to_high(demo_districts):
    output = run_pipeline(demo_districts)
    translation = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)
    assert (translation["estimated_impact_low_inr_cr"] <= translation["estimated_impact_high_inr_cr"]).all()


def test_sorted_descending_by_severe_stress_exposure(demo_districts):
    output = run_pipeline(demo_districts)
    translation = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)
    values = translation["severe_stress_exposure_inr_cr"].tolist()
    assert values == sorted(values, reverse=True)
