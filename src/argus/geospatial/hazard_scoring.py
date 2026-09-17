"""Composite Physical Hazard Score — Enterprise build.

Combines FOUR independently-sourced, real hazard signals into one 0-100
district-level score:

  flood_component    = 0.6 x satellite NDWI flood-extent score (Proxy — synthetic
                        tile, see geospatial/flood_index.py's own docstring) +
                        0.4 x IMD-rainfall-derived flood proxy (Derived from
                        Observed IMD sub-divisional rainfall; see ingestion/rainfall_imd.py)
  drought_component   = IMD-rainfall-derived drought proxy (Derived, same source)
  cyclone_component   = IMD best-track-derived cyclone exposure (Derived from
                        Observed IMD RSMC New Delhi records; see ingestion/cyclone_imd.py)
  heat_component      = IMD station climatological extreme-heat exposure (Derived
                        from Observed IMD station normals; see ingestion/heat_imd.py)

  composite_hazard_score = weighted sum of the four components (weights in
  common/config.py, sum to 1.0)

This is still a transparent, linear, weighted-index methodology — deliberately, so
that explainability/feature_contributions.py can decompose a score exactly rather
than approximate it. What changed from the V1 demo build is that three of these four
signals are now Derived from real, cited, government-published historical records
(IMD cyclone best-track, IMD rainfall, IMD station climatology) instead of an
illustrative fixture — see docs/data-sources.md and docs/evaluation-report.md for
the full retrieval notes, and common/provenance.py for what every returned
DataProvenance record actually certifies.
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import HAZARD_WEIGHTS, HIGH_RISK_THRESHOLD
from argus.common.provenance import DataProvenance, DataStatus
from argus.geospatial.flood_index import flood_extent_score
from argus.ingestion.copernicus import load_district_tiles
from argus.ingestion.cyclone_imd import compute_district_cyclone_exposure
from argus.ingestion.heat_imd import compute_district_heat_exposure
from argus.ingestion.rainfall_imd import compute_district_rainfall_hazards

# The satellite NDWI signal (0.6 weight inside flood_component) is a synthetic,
# deterministically-generated stand-in tile (see ingestion/copernicus.py's own
# docstring) rather than a live Sentinel-2/Copernicus EMS pull this sandbox cannot
# reach -- so, unlike the other three hazard components, it carries a fixed PROXY
# provenance record rather than one computed per-run. This is what the Data
# Lineage page shows next to the flood component's satellite contribution so it is
# never mistaken for an Observed measurement.
SATELLITE_FLOOD_PROVENANCE = DataProvenance(
    dataset="Synthetic Sentinel-2-style green/NIR reflectance tile",
    source_organization="Argus (generated locally — no live Copernicus EMS/Sentinel-2 feed reachable in this build)",
    source_url="src/argus/ingestion/copernicus.py",
    publication_date="n/a — not a published dataset",
    observation_period="n/a",
    geographic_resolution="district (synthetic per-district tile)",
    unit="fraction of pixels NDWI-positive (0-1)",
    methodology=(
        "fetch_live_tile() returns None in this sandbox (Copernicus EMS unreachable); "
        "synthetic_tile() deterministically generates a green/NIR tile from the "
        "bundled true_water_fraction label so the real NDWI computation "
        "(geospatial/flood_index.py) still runs against real array math and a known "
        "ground truth. Swap fetch_live_tile() for an authenticated Sentinel-2/Copernicus "
        "EMS download in production; nothing downstream changes."
    ),
    status=DataStatus.PROXY,
    retrieved_at=DataProvenance.now_iso(),
)


def compute_hazard_scores(districts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, DataProvenance]]:
    """districts needs 'district', 'state', 'lat', 'lon'. Returns (scores,
    provenance_by_source) -- the provenance dict is what the Data Lineage page reads
    to cite every component."""
    rainfall, rainfall_prov = compute_district_rainfall_hazards(districts)
    cyclone, cyclone_prov = compute_district_cyclone_exposure(districts)
    heat, heat_prov = compute_district_heat_exposure(districts[["district", "state"]])
    tiles = load_district_tiles()

    df = districts.merge(rainfall, on="district", how="left")
    df = df.merge(cyclone, on="district", how="left", suffixes=("", "_cyclone"))
    df = df.merge(heat, on="district", how="left", suffixes=("", "_heat"))

    rows = []
    for _, r in df.iterrows():
        district = r["district"]
        green, nir = tiles.get(district, (None, None))
        # green/nir are always paired by construction (see load_district_tiles) — both
        # None or both arrays, never mixed — so checking both narrows nir's type for
        # mypy too, not just a runtime nicety.
        satellite_score = flood_extent_score(green, nir) if green is not None and nir is not None else 0.0

        flood_component = 0.6 * satellite_score + 0.4 * float(r["flood_risk"])
        drought_component = float(r["drought_risk"])
        cyclone_component = float(r["cyclone_risk"])
        heat_component = float(r["heat_risk"])

        composite = (
            HAZARD_WEIGHTS["flood"] * flood_component
            + HAZARD_WEIGHTS["drought"] * drought_component
            + HAZARD_WEIGHTS["cyclone"] * cyclone_component
            + HAZARD_WEIGHTS["heat"] * heat_component
        )
        rows.append(
            {
                "district": district,
                "state": r.get("state"),
                "satellite_flood_extent_score": round(satellite_score, 2),
                "flood_component": round(flood_component, 2),
                "drought_component": round(drought_component, 2),
                "cyclone_component": round(cyclone_component, 2),
                "heat_component": round(heat_component, 2),
                "composite_hazard_score": round(composite, 2),
                "is_high_risk": composite >= HIGH_RISK_THRESHOLD,
                "rainfall_anomaly_pct": r.get("rainfall_anomaly_pct"),
                "cyclone_storm_count": r.get("storm_count"),
                "heat_peak_month_extreme_c": r.get("peak_month_extreme_high_c"),
                "heat_data_status": r.get("status"),
            }
        )
    scores = pd.DataFrame(rows)
    provenance = {
        "rainfall": rainfall_prov,
        "cyclone": cyclone_prov,
        "heat": heat_prov,
        "satellite_flood": SATELLITE_FLOOD_PROVENANCE,
    }
    return scores, provenance


if __name__ == "__main__":
    from argus.ingestion.base import load_sample_csv

    demo = load_sample_csv("district_hazard_inputs.csv")[["district", "state", "lat", "lon"]]
    result, _ = compute_hazard_scores(demo)
    print(result.to_string(index=False))
