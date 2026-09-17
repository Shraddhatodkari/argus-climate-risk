"""Unit tests: every real-data ingestion module's offline fallback path (the frozen
local snapshot each module falls back to when a live fetch isn't available — see
each module's own docstring for why this sandboxed environment always takes that
path) plus the RBI macro-context module retained from the V1 demo build."""

import pandas as pd

from argus.ingestion import cyclone_imd, financial_exposure, heat_imd, rainfall_imd, rbi

DEMO_DISTRICTS = pd.DataFrame(
    [
        {"district": "Cuttack", "state": "Odisha", "lat": 20.4625, "lon": 85.8828},
        {"district": "Latur", "state": "Maharashtra", "lat": 18.4088, "lon": 76.5604},
    ]
)


def test_cyclone_imd_returns_a_score_and_provenance_for_every_district():
    result, provenance = cyclone_imd.compute_district_cyclone_exposure(DEMO_DISTRICTS[["district", "lat", "lon"]])
    assert set(result["district"]) == {"Cuttack", "Latur"}
    assert result["cyclone_risk"].between(0, 100).all()
    assert provenance.source_organization.startswith("India Meteorological Department")


def test_rainfall_imd_returns_flood_and_drought_proxies():
    result, provenance = rainfall_imd.compute_district_rainfall_hazards(DEMO_DISTRICTS[["district"]])
    assert {"flood_risk", "drought_risk", "rainfall_anomaly_pct"}.issubset(result.columns)
    assert len(result) == 2
    assert provenance.geographic_resolution.startswith("IMD meteorological subdivision")


def test_heat_imd_returns_a_score_for_every_district():
    result, _provenance = heat_imd.compute_district_heat_exposure(DEMO_DISTRICTS[["district", "state"]])
    assert len(result) == 2
    assert result["heat_risk"].between(0, 100).all()
    assert (result["status"] == "Observed").all()  # neither demo district needs the Kanchipuram/Chennai proxy


def test_heat_imd_labels_the_proxy_district_correctly():
    kanchipuram = pd.DataFrame([{"district": "Kanchipuram", "state": "Tamil Nadu"}])
    result, _ = heat_imd.compute_district_heat_exposure(kanchipuram)
    assert result.iloc[0]["status"] == "Proxy"
    assert "CHENNAI" in result.iloc[0]["station_used"]


def test_financial_exposure_never_exceeds_the_state_total():
    result, provenance = financial_exposure.load_public_exposure(DEMO_DISTRICTS[["district", "state"]])
    assert (result["public_exposure_inr_cr"] <= result["state_credit_potential_inr_cr"]).all()
    assert provenance.status.value == "Derived"


def test_financial_exposure_institution_portfolio_hook_returns_none_by_default():
    assert financial_exposure.load_institution_portfolio() is None


def test_rbi_macro_context_fallback_returns_a_dataframe():
    df = rbi.load_macro_context()
    assert {"series", "value"}.issubset(df.columns)
    assert len(df) > 0
