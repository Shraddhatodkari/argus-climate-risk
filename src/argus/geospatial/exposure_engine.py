"""Climate-Adjusted Exposure at Risk — Enterprise build.

Wires the full Hazard -> Exposure -> Vulnerability -> Physical Impact chain into one
call, and returns every intermediate table (not just the final flat one) so the
Data Lineage page, the Stress Testing tab, and the Concentration Analysis tab can
all be built from the SAME pipeline run rather than each recomputing it slightly
differently. This is the one place that composes:

    geospatial.hazard_scoring        -- 4 real hazard components per district
    ingestion.financial_exposure     -- public exposure estimate per district
    financial.sector_allocation      -- exposure split into Agriculture/MSME/Other
    financial.impact_engine          -- Hazard x Exposure x Vulnerability

into a ``PipelineOutput`` bundle, plus a flat backward-compatible
``exposure_at_risk`` table (district-level, for the main Exposure Table view and
the high-risk review-queue flag) built by rolling the sector-level detail up to one
row per district.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from argus.common.provenance import DataProvenance
from argus.common.schemas import ExposureSchema
from argus.financial.impact_engine import compute_physical_impact, rollup_by_district
from argus.financial.sector_allocation import allocate_sectors
from argus.geospatial.hazard_scoring import compute_hazard_scores
from argus.ingestion.financial_exposure import load_public_exposure


@dataclass(frozen=True)
class PipelineOutput:
    districts: pd.DataFrame
    hazard_scores: pd.DataFrame
    public_exposure: pd.DataFrame
    sector_exposure: pd.DataFrame
    impact_detail: pd.DataFrame
    impact_by_district: pd.DataFrame
    exposure_at_risk: pd.DataFrame
    provenance: dict[str, DataProvenance]


def run_pipeline(districts: pd.DataFrame) -> PipelineOutput:
    """districts needs district, state, lat, lon (the district registry — see
    common/schemas.py's DistrictHazardInputSchema)."""
    hazard_scores, hazard_provenance = compute_hazard_scores(districts)
    public_exposure, exposure_provenance = load_public_exposure(districts[["district", "state"]])
    sector_exposure, sector_provenance = allocate_sectors(public_exposure)

    impact_detail = compute_physical_impact(hazard_scores, sector_exposure)
    impact_by_district = rollup_by_district(impact_detail)

    flat = hazard_scores.merge(public_exposure[["district", "public_exposure_inr_cr"]], on="district", how="left")
    flat = flat.merge(
        impact_by_district[["district", "physical_impact_inr_cr", "primary_hazard", "most_affected_sector"]],
        on="district",
        how="left",
    )
    flat = flat.rename(columns={"physical_impact_inr_cr": "climate_exposed_exposure_inr_cr"})
    exposure_at_risk = flat[
        [
            "district",
            "state",
            "flood_component",
            "drought_component",
            "cyclone_component",
            "heat_component",
            "composite_hazard_score",
            "public_exposure_inr_cr",
            "climate_exposed_exposure_inr_cr",
            "primary_hazard",
            "most_affected_sector",
            "is_high_risk",
        ]
    ].copy()
    ExposureSchema.validate(exposure_at_risk)
    exposure_at_risk = exposure_at_risk.sort_values("climate_exposed_exposure_inr_cr", ascending=False).reset_index(
        drop=True
    )

    provenance = {**hazard_provenance, "financial_exposure": exposure_provenance, "sector_allocation": sector_provenance}
    return PipelineOutput(
        districts=districts,
        hazard_scores=hazard_scores,
        public_exposure=public_exposure,
        sector_exposure=sector_exposure,
        impact_detail=impact_detail,
        impact_by_district=impact_by_district,
        exposure_at_risk=exposure_at_risk,
        provenance=provenance,
    )


def compute_exposure_at_risk(districts: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible flat entry point — the table most callers (scoring_agent,
    narrative_agent, orchestrator, the Streamlit exposure table) actually want."""
    return run_pipeline(districts).exposure_at_risk


if __name__ == "__main__":
    from argus.ingestion.base import load_sample_csv

    demo = load_sample_csv("district_hazard_inputs.csv")[["district", "state", "lat", "lon"]]
    print(compute_exposure_at_risk(demo).to_string(index=False))
