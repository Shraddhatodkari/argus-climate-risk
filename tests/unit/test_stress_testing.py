"""Unit tests: climate stress testing (Baseline / Moderate Stress / Severe Stress /
Long-Term Scenario) — item #5 of the Enterprise spec."""

import pandas as pd
import pytest

from argus.common.config import HAZARD_WEIGHTS, IMPACT_RATE_RANGE, STRESS_SCENARIOS
from argus.financial.stress_testing import (
    district_scenario_summary,
    run_all_scenarios,
    run_stress_scenario,
)


@pytest.fixture
def hazard_scores():
    return pd.DataFrame(
        [
            {
                "district": "Cuttack",
                "flood_component": 22.05,
                "drought_component": 14.53,
                "cyclone_component": 88.08,
                "heat_component": 67.22,
            }
        ]
    )


@pytest.fixture
def sector_exposure():
    return pd.DataFrame(
        [
            {"district": "Cuttack", "state": "Odisha", "sector": "Agriculture", "sector_exposure_inr_cr": 10241.0},
            {"district": "Cuttack", "state": "Odisha", "sector": "MSME", "sector_exposure_inr_cr": 7681.0},
        ]
    )


def test_unknown_scenario_raises(hazard_scores, sector_exposure):
    with pytest.raises(ValueError):
        run_stress_scenario("Not A Real Scenario", hazard_scores, sector_exposure)


def test_scenario_detail_has_one_row_per_district_sector_hazard(hazard_scores, sector_exposure):
    detail = run_stress_scenario("Baseline", hazard_scores, sector_exposure)
    assert len(detail) == 2 * 4  # 2 sectors x 4 hazards
    assert set(detail["hazard"]) == {"flood", "drought", "cyclone", "heat"}


def test_stressed_score_never_exceeds_100(hazard_scores, sector_exposure):
    detail = run_stress_scenario("Long-Term Scenario", hazard_scores, sector_exposure)
    assert (detail["stressed_hazard_score"] <= 100.0).all()


def test_stressed_exposure_never_exceeds_the_affected_exposure_base(hazard_scores, sector_exposure):
    """The same bounded-formula guarantee as impact_engine.py: a stressed financial
    exposure can never exceed the exposure_affected base it's computed from, even at
    Long-Term-Scenario severity (the largest hazard_multiplier)."""
    for scenario in STRESS_SCENARIOS:
        detail = run_stress_scenario(scenario, hazard_scores, sector_exposure)
        assert (detail["stressed_financial_exposure_inr_cr"] <= detail["exposure_affected_inr_cr"] + 0.01).all()


def test_impact_range_uses_the_disclosed_rate_range(hazard_scores, sector_exposure):
    detail = run_stress_scenario("Severe Stress", hazard_scores, sector_exposure)
    low, high = IMPACT_RATE_RANGE
    for _, row in detail.iterrows():
        assert abs(row["estimated_impact_low_inr_cr"] - row["stressed_financial_exposure_inr_cr"] * low) < 0.01
        assert abs(row["estimated_impact_high_inr_cr"] - row["stressed_financial_exposure_inr_cr"] * high) < 0.01
        assert row["estimated_impact_low_inr_cr"] <= row["estimated_impact_high_inr_cr"]


def test_more_severe_scenarios_never_produce_less_stressed_exposure(hazard_scores, sector_exposure):
    """Baseline -> Moderate -> Severe -> Long-Term should be monotonically
    non-decreasing in total stressed exposure for a fixed hazard/sector profile."""
    detail = run_all_scenarios(hazard_scores, sector_exposure)
    summary = district_scenario_summary(detail)
    totals = [
        summary[summary["scenario"] == name]["stressed_financial_exposure_inr_cr"].sum()
        for name in STRESS_SCENARIOS
    ]
    assert totals == sorted(totals)


def test_run_all_scenarios_covers_every_declared_scenario(hazard_scores, sector_exposure):
    detail = run_all_scenarios(hazard_scores, sector_exposure)
    assert set(detail["scenario"]) == set(STRESS_SCENARIOS.keys())


def test_district_scenario_summary_handles_empty_input():
    empty = pd.DataFrame(columns=["scenario", "district", "state", "stressed_financial_exposure_inr_cr"])
    assert district_scenario_summary(empty).empty


def test_hazard_weights_still_sum_to_one_for_this_module_to_stay_bounded():
    # Sanity check tying this module's bounding guarantee to the shared constant.
    assert abs(sum(HAZARD_WEIGHTS.values()) - 1.0) < 1e-9
