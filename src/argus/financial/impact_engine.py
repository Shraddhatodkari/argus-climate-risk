"""Physical Climate Impact = Hazard x Exposure x Vulnerability.

This is the core Enterprise-build formula, replacing the V1 demo's flatter
"exposure_at_risk = hazard% x total credit" calculation with a sector-aware one:
every district's exposure is first split by sector (financial/sector_allocation.py),
then each sector's exposure is weighted by how sensitive that sector actually is to
each of the four hazard components (financial/vulnerability.py) -- not the district's
one blended hazard score applied uniformly to its whole economy.

    combined_sensitivity(district, sector)
        = sum over hazard in {flood, drought, cyclone, heat} of:
              HAZARD_WEIGHTS[hazard] x (hazard_component_score / 100)
              x vulnerability_weight(sector, hazard)

    physical_impact_inr_cr(district, sector)
        = combined_sensitivity(district, sector) x sector_exposure_inr_cr(district, sector)

Reusing HAZARD_WEIGHTS (the same weights the composite hazard score uses, summing to
1.0) rather than summing four independent (score x weight) terms uncapped is a
deliberate correctness choice, not just a style preference: since every
hazard_component_score is in [0, 100] and every vulnerability_weight is in [0, 1],
combined_sensitivity is guaranteed to stay in [0, 1] -- so physical_impact_inr_cr can
never exceed a sector's actual exposure, which an unweighted sum across four
"independent" hazard contributions could (and did, in an earlier draft of this
module: a sector highly vulnerable to several severe hazards at once could show more
climate-exposed exposure than it had exposure to begin with, which makes no sense to
a risk officer reading the number). The formula stays linear and fully decomposable
(same design principle as geospatial/hazard_scoring.py's composite score) --
explainability/feature_contributions.py-style auditability matters more here than a
marginally more sophisticated non-linear model this project has no loss data to
validate against.

This module is a pure function of the deterministic engine's own outputs (hazard
scores, sector exposure, vulnerability weights) -- no LLM call anywhere in this file.
See docs/architecture.md's "deterministic engine vs LLM" section: this is exactly
the boundary it describes.
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import HAZARD_WEIGHTS
from argus.financial.vulnerability import HAZARDS, vulnerability_weight

HAZARD_COMPONENT_COLUMN = {
    "flood": "flood_component",
    "drought": "drought_component",
    "cyclone": "cyclone_component",
    "heat": "heat_component",
}


def compute_physical_impact(hazard_scores: pd.DataFrame, sector_exposure: pd.DataFrame) -> pd.DataFrame:
    """hazard_scores: one row per district, with flood_component/drought_component/
    cyclone_component/heat_component (0-100) -- the output of
    geospatial.hazard_scoring.compute_hazard_scores(). sector_exposure: one row per
    (district, sector), with sector_exposure_inr_cr -- the output of
    financial.sector_allocation.allocate_sectors(). Returns one row per
    (district, sector, hazard) with the exact contribution, PLUS the
    (district, sector) rollup as a second-level aggregate the caller can pivot on."""
    hz = hazard_scores.set_index("district")
    rows = []
    for _, se in sector_exposure.iterrows():
        district, sector = se["district"], se["sector"]
        if district not in hz.index:
            continue
        hz_row = hz.loc[district]
        exposure = float(se["sector_exposure_inr_cr"])
        for hazard in HAZARDS:
            score = float(hz_row[HAZARD_COMPONENT_COLUMN[hazard]])
            weight = vulnerability_weight(sector, hazard)
            # This hazard's share of combined_sensitivity -- see the module
            # docstring for why HAZARD_WEIGHTS[hazard] is folded in here rather than
            # summing four uncapped (score x weight) terms.
            sensitivity_share = HAZARD_WEIGHTS[hazard] * (score / 100.0) * weight
            contribution = sensitivity_share * exposure
            rows.append(
                {
                    "district": district,
                    "state": se["state"],
                    "sector": sector,
                    "hazard": hazard,
                    "hazard_score": round(score, 2),
                    "hazard_weight": HAZARD_WEIGHTS[hazard],
                    "vulnerability_weight": weight,
                    "sector_exposure_inr_cr": exposure,
                    "physical_impact_inr_cr": round(contribution, 2),
                }
            )
    return pd.DataFrame(rows)


def rollup_by_district_sector(impact_detail: pd.DataFrame) -> pd.DataFrame:
    """Sums each (district, sector)'s four hazard contributions -- the level a risk
    officer actually reads ("Latur / Agriculture: Rs X cr physical climate impact,
    driven mostly by drought"), plus which single hazard contributed the most."""
    if impact_detail.empty:
        return impact_detail
    grouped = impact_detail.groupby(["district", "state", "sector"], as_index=False).agg(
        physical_impact_inr_cr=("physical_impact_inr_cr", "sum"),
        sector_exposure_inr_cr=("sector_exposure_inr_cr", "first"),
    )
    idx = impact_detail.groupby(["district", "sector"])["physical_impact_inr_cr"].idxmax()
    primary = impact_detail.loc[idx, ["district", "sector", "hazard"]].rename(columns={"hazard": "primary_hazard"})
    return grouped.merge(primary, on=["district", "sector"], how="left")


def rollup_by_district(impact_detail: pd.DataFrame) -> pd.DataFrame:
    """District-level total physical impact across all sectors -- the number that
    belongs on the exposure table next to composite_hazard_score."""
    if impact_detail.empty:
        return impact_detail
    grouped = impact_detail.groupby(["district", "state"], as_index=False).agg(
        physical_impact_inr_cr=("physical_impact_inr_cr", "sum"),
    )
    # total_exposure_inr_cr needs the *unique* per-sector exposure summed once, not
    # once per hazard row (impact_detail has 4 hazard rows per district/sector) --
    # computed separately rather than folded into the agg() above for clarity.
    exposure_by_district = (
        impact_detail.drop_duplicates(subset=["district", "sector"])
        .groupby("district", as_index=False)["sector_exposure_inr_cr"]
        .sum()
        .rename(columns={"sector_exposure_inr_cr": "total_exposure_inr_cr"})
    )
    grouped = grouped.merge(exposure_by_district, on="district", how="left")

    idx = impact_detail.groupby(["district", "hazard"])["physical_impact_inr_cr"].sum().groupby("district").idxmax()
    primary_hazard = pd.Series({d: h for d, h in idx.values}, name="primary_hazard")
    grouped = grouped.merge(primary_hazard.rename_axis("district").reset_index(), on="district", how="left")

    sector_totals = impact_detail.groupby(["district", "sector"])["physical_impact_inr_cr"].sum()
    most_affected = sector_totals.groupby("district").idxmax().apply(lambda t: t[1])
    grouped = grouped.merge(
        most_affected.rename("most_affected_sector").rename_axis("district").reset_index(), on="district", how="left"
    )
    return grouped


if __name__ == "__main__":
    import pandas as pd

    hz = pd.DataFrame(
        [{"district": "Latur", "flood_component": 8.4, "drought_component": 15.38, "cyclone_component": 9.6, "heat_component": 67.22}]
    )
    se = pd.DataFrame(
        [
            {"district": "Latur", "state": "Maharashtra", "sector": "Agriculture", "sector_exposure_inr_cr": 5075.0},
            {"district": "Latur", "state": "Maharashtra", "sector": "MSME", "sector_exposure_inr_cr": 14572.0},
            {"district": "Latur", "state": "Maharashtra", "sector": "Other", "sector_exposure_inr_cr": 2807.0},
        ]
    )
    detail = compute_physical_impact(hz, se)
    print(detail.to_string(index=False))
    print(rollup_by_district_sector(detail).to_string(index=False))
    print(rollup_by_district(detail).to_string(index=False))
