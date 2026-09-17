"""Financial-risk translation -- turns the Hazard -> Exposure -> Vulnerability chain
into the summary table a risk committee actually reads:

    Portfolio exposure                 Rs X Cr   (public-data estimate; institution
                                                    portfolio if one is configured)
    Climate-exposed exposure           Rs Y Cr   (baseline-scenario physical impact)
    Severe-stress exposure             Rs Z Cr   (Severe Stress scenario, stressed
                                                    financial exposure)
    Estimated impact range             Rs A-B Cr (Severe Stress scenario's impact range)
    Primary hazard                     <hazard driving the most impact>
    Most affected sector               <sector with the largest physical impact>

Every figure in the row this module returns can be traced back to exactly one of:
geospatial.hazard_scoring (hazard scores), financial.sector_allocation (exposure
split), financial.vulnerability (sensitivity weights), financial.impact_engine
(baseline physical impact) or financial.stress_testing (the Severe Stress scenario)
-- that chain of function calls IS the methodology note the Data Lineage page shows
for this row.

Honesty note on a non-obvious interaction: ``climate_exposed_exposure_inr_cr``
("Baseline") applies the district's UNSTRESSED hazard score against its FULL sector
exposure, while ``severe_stress_exposure_inr_cr`` applies a stressed (up to 1.6x,
capped at 100) hazard score against only the Severe Stress scenario's
``exposure_affected_share`` (0.75) of that same exposure. For a district whose
unstressed hazard score is already close to saturating the 0-100 scale (e.g. Puri's
cyclone_component ~88.9), the capped multiplier adds little headroom, and the
smaller 75%-of-exposure base can make severe_stress_exposure_inr_cr come out LOWER
than climate_exposed_exposure_inr_cr for that district -- confirmed in this build's
own demo data (see tests/unit/test_financial_translation.py) and NOT a bug: the two
figures answer genuinely different questions ("today's expected physical impact
across the whole exposed sector" vs. "what a smaller, severely-stressed slice of
that exposure could look like"), not "before vs. after" on the same base. A
production Financial Translation view should say so next to the two figures rather
than let a reader assume Severe Stress is always the larger number.
"""

from __future__ import annotations

import pandas as pd

from argus.financial.impact_engine import compute_physical_impact, rollup_by_district
from argus.financial.stress_testing import district_scenario_summary, run_stress_scenario


def build_financial_translation(
    hazard_scores: pd.DataFrame, sector_exposure: pd.DataFrame, district_exposure: pd.DataFrame
) -> pd.DataFrame:
    """district_exposure needs 'district', 'public_exposure_inr_cr' (the portfolio
    exposure row -- swap in an institution portfolio's district totals here once one
    is configured; see ingestion.financial_exposure.load_institution_portfolio)."""
    baseline_detail = compute_physical_impact(hazard_scores, sector_exposure)
    baseline_rollup = rollup_by_district(baseline_detail)

    severe_detail = run_stress_scenario("Severe Stress", hazard_scores, sector_exposure)
    severe_summary = district_scenario_summary(severe_detail)

    rows = []
    for _, d in district_exposure.iterrows():
        district = d["district"]
        base_row = baseline_rollup[baseline_rollup["district"] == district]
        severe_row = severe_summary[severe_summary["district"] == district]
        if base_row.empty or severe_row.empty:
            continue
        base_row = base_row.iloc[0]
        severe_row = severe_row.iloc[0]
        rows.append(
            {
                "district": district,
                "state": d.get("state"),
                "portfolio_exposure_inr_cr": round(float(d["public_exposure_inr_cr"]), 1),
                "portfolio_exposure_source": d.get("exposure_source", "Public-data estimate (see financial_exposure.py)"),
                "climate_exposed_exposure_inr_cr": round(float(base_row["physical_impact_inr_cr"]), 1),
                "severe_stress_exposure_inr_cr": round(float(severe_row["stressed_financial_exposure_inr_cr"]), 1),
                "estimated_impact_low_inr_cr": round(float(severe_row["estimated_impact_low_inr_cr"]), 1),
                "estimated_impact_high_inr_cr": round(float(severe_row["estimated_impact_high_inr_cr"]), 1),
                "primary_hazard": base_row["primary_hazard"],
                "most_affected_sector": base_row["most_affected_sector"],
            }
        )
    return pd.DataFrame(rows).sort_values("severe_stress_exposure_inr_cr", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from argus.geospatial.exposure_engine import run_pipeline
    from argus.ingestion.base import load_sample_csv

    demo = load_sample_csv("district_hazard_inputs.csv")[["district", "state", "lat", "lon"]]
    output = run_pipeline(demo)
    result = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)
    print(result.to_string(index=False))
