"""Unit tests: concentration analysis (State -> District -> Sector -> Hazard ->
Exposure rollups) — item #7 of the Enterprise spec."""

import pandas as pd
import pytest

from argus.analytics.concentration import (
    concentration_by_hazard,
    concentration_by_sector,
    concentration_by_state,
    concentration_by_state_hazard,
    sector_geography_insight,
    top_concentration_insight,
)


@pytest.fixture
def impact_detail():
    return pd.DataFrame(
        [
            {"district": "Cuttack", "state": "Odisha", "sector": "Agriculture", "hazard": "cyclone", "physical_impact_inr_cr": 3000.0},
            {"district": "Cuttack", "state": "Odisha", "sector": "MSME", "hazard": "flood", "physical_impact_inr_cr": 500.0},
            {"district": "Puri", "state": "Odisha", "sector": "Agriculture", "hazard": "cyclone", "physical_impact_inr_cr": 2000.0},
            {"district": "Latur", "state": "Maharashtra", "sector": "Agriculture", "hazard": "drought", "physical_impact_inr_cr": 1000.0},
        ]
    )


def test_concentration_by_state_sums_and_sorts_descending(impact_detail):
    result = concentration_by_state(impact_detail)
    assert list(result["state"]) == ["Odisha", "Maharashtra"]
    assert result.iloc[0]["physical_impact_inr_cr"] == pytest.approx(5500.0)


def test_concentration_by_hazard_sums_and_sorts_descending(impact_detail):
    result = concentration_by_hazard(impact_detail)
    assert result.iloc[0]["hazard"] == "cyclone"
    assert result.iloc[0]["physical_impact_inr_cr"] == pytest.approx(5000.0)


def test_concentration_by_sector_sums_and_sorts_descending(impact_detail):
    result = concentration_by_sector(impact_detail)
    assert result.iloc[0]["sector"] == "Agriculture"
    assert result.iloc[0]["physical_impact_inr_cr"] == pytest.approx(6000.0)


def test_concentration_by_state_hazard_cross_cut(impact_detail):
    result = concentration_by_state_hazard(impact_detail)
    top = result.iloc[0]
    assert top["state"] == "Odisha"
    assert top["hazard"] == "cyclone"
    assert top["physical_impact_inr_cr"] == pytest.approx(5000.0)


def test_top_concentration_insight_names_the_largest_state_hazard_pair(impact_detail):
    sentence = top_concentration_insight(impact_detail)
    assert "Odisha" in sentence
    assert "cyclone" in sentence
    total = impact_detail["physical_impact_inr_cr"].sum()
    expected_pct = round(100.0 * 5000.0 / total)
    assert f"{expected_pct}%" in sentence


def test_sector_geography_insight_names_top_sector_state_and_hazard(impact_detail):
    sentence = sector_geography_insight(impact_detail)
    assert "Agriculture" in sentence
    assert "Odisha" in sentence
    assert "cyclone" in sentence


def test_empty_impact_detail_produces_a_safe_message_not_a_crash():
    empty = pd.DataFrame(columns=["district", "state", "sector", "hazard", "physical_impact_inr_cr"])
    assert concentration_by_state(empty).empty
    assert "No physical impact data" in top_concentration_insight(empty)
    assert "No physical impact data" in sector_geography_insight(empty)
