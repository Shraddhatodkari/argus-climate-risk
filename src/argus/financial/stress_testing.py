"""Climate stress testing — Baseline / Moderate Stress / Severe Stress / Long-Term
Scenario, each running the same deterministic chain:

    Hazard shock -> Exposure affected -> Vulnerability -> Stressed financial exposure
    -> Estimated impact range

Every number this module returns is a MODEL ESTIMATE under disclosed assumptions
(common/config.py's STRESS_SCENARIOS and IMPACT_RATE_RANGE), not a claim about an
institution's actual or projected losses -- every caller (the Streamlit stress-test
tab, the financial translation summary) must keep that framing next to the numbers,
and this module's own docstrings say so at every step so nothing downstream can
quietly drop the caveat.

Per scenario, per (district, sector, hazard):
1. Hazard shock:      stressed_hazard_score = min(100, hazard_score x hazard_multiplier)
2. Exposure affected:  exposure_affected_inr_cr = sector_exposure_inr_cr x exposure_affected_share
                       (a worse scenario assumes a larger share of the district's
                       sector exposure sits inside the hazard's affected footprint --
                       a disclosed modelling choice, not a measured fact)
3. Vulnerability:      the same sector/hazard sensitivity weight used in impact_engine.py,
                       combined with HAZARD_WEIGHTS the same way (see impact_engine.py's
                       docstring) so a stressed financial exposure can never exceed the
                       exposure-affected base, even at Long-Term-Scenario severity
4. Stressed financial exposure = exposure_affected_inr_cr x HAZARD_WEIGHTS[hazard]
                       x vulnerability_weight x (stressed_hazard_score / 100)
5. Estimated impact range = stressed financial exposure x IMPACT_RATE_RANGE
                       (a disclosed low/high loss-given-hazard rate, giving a range
                       rather than a false-precision point estimate)
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import HAZARD_WEIGHTS, IMPACT_RATE_RANGE, STRESS_SCENARIOS
from argus.financial.impact_engine import HAZARD_COMPONENT_COLUMN
from argus.financial.vulnerability import HAZARDS, vulnerability_weight


def run_stress_scenario(scenario_name: str, hazard_scores: pd.DataFrame, sector_exposure: pd.DataFrame) -> pd.DataFrame:
    if scenario_name not in STRESS_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario_name!r}; expected one of {list(STRESS_SCENARIOS)}")
    params = STRESS_SCENARIOS[scenario_name]
    hz = hazard_scores.set_index("district")

    rows = []
    for _, se in sector_exposure.iterrows():
        district, sector = se["district"], se["sector"]
        if district not in hz.index:
            continue
        hz_row = hz.loc[district]
        exposure_affected = float(se["sector_exposure_inr_cr"]) * params["exposure_affected_share"]
        for hazard in HAZARDS:
            base_score = float(hz_row[HAZARD_COMPONENT_COLUMN[hazard]])
            stressed_score = min(100.0, base_score * params["hazard_multiplier"])
            weight = vulnerability_weight(sector, hazard)
            stressed_exposure = exposure_affected * HAZARD_WEIGHTS[hazard] * weight * (stressed_score / 100.0)
            low, high = IMPACT_RATE_RANGE
            rows.append(
                {
                    "scenario": scenario_name,
                    "district": district,
                    "state": se["state"],
                    "sector": sector,
                    "hazard": hazard,
                    "base_hazard_score": round(base_score, 2),
                    "stressed_hazard_score": round(stressed_score, 2),
                    "exposure_affected_inr_cr": round(exposure_affected, 1),
                    "vulnerability_weight": weight,
                    "stressed_financial_exposure_inr_cr": round(stressed_exposure, 2),
                    "estimated_impact_low_inr_cr": round(stressed_exposure * low, 2),
                    "estimated_impact_high_inr_cr": round(stressed_exposure * high, 2),
                }
            )
    return pd.DataFrame(rows)


def run_all_scenarios(hazard_scores: pd.DataFrame, sector_exposure: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [run_stress_scenario(name, hazard_scores, sector_exposure) for name in STRESS_SCENARIOS], ignore_index=True
    )


def district_scenario_summary(stress_detail: pd.DataFrame) -> pd.DataFrame:
    """Rolls the detail table up to one row per (scenario, district) -- what the
    Stress Testing tab's main table actually shows."""
    if stress_detail.empty:
        return stress_detail
    return (
        stress_detail.groupby(["scenario", "district", "state"], as_index=False)
        .agg(
            stressed_financial_exposure_inr_cr=("stressed_financial_exposure_inr_cr", "sum"),
            estimated_impact_low_inr_cr=("estimated_impact_low_inr_cr", "sum"),
            estimated_impact_high_inr_cr=("estimated_impact_high_inr_cr", "sum"),
        )
        .round(1)
    )


if __name__ == "__main__":
    hz = pd.DataFrame(
        [{"district": "Cuttack", "flood_component": 22.05, "drought_component": 14.53, "cyclone_component": 88.08, "heat_component": 67.22}]
    )
    se = pd.DataFrame(
        [
            {"district": "Cuttack", "state": "Odisha", "sector": "Agriculture", "sector_exposure_inr_cr": 10241.0},
            {"district": "Cuttack", "state": "Odisha", "sector": "MSME", "sector_exposure_inr_cr": 7681.0},
        ]
    )
    detail = run_all_scenarios(hz, se)
    print(district_scenario_summary(detail).to_string(index=False))
