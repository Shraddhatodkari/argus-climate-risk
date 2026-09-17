"""Asset/location-level geospatial intelligence — Enterprise build, item #3.

The district-level composite score ("Cuttack = 46.1") is useful for portfolio-level
screening, but a real risk team eventually needs the finer-grained chain:

    Borrower/asset -> Latitude/longitude -> District -> Flood exposure ->
    Drought exposure -> Cyclone exposure -> Heat exposure -> Sector vulnerability
    -> Asset-level physical climate impact

This module builds exactly that chain, reusing the same district-level hazard
components (geospatial/hazard_scoring.py) and the same sector/hazard vulnerability
weights (financial/vulnerability.py, financial/config.py's SECTOR_VULNERABILITY) that
the district/sector-level pipeline already uses — an asset's physical impact is
computed with the identical formula as impact_engine.py's sector-level one, just
applied to one asset's own exposure instead of a whole sector's.

Honesty note on the data itself: no real, individually-geolocated borrower/asset
dataset is publicly available (and could not be — that is confidential loan-book
data by definition, real individual borrower locations are not something a public,
non-proprietary project can ship). data/samples/asset_locations.csv is therefore
explicitly ILLUSTRATIVE sample data (every row's is_synthetic="true", asset names
prefixed "Illustrative...") showing the SHAPE an institution's real asset/borrower
file would take, placed at plausible points around each demo district's real
centroid. A deploying institution replaces this one file with their own real,
geolocated book (same required columns: asset_id, sector, district, lat, lon,
asset_exposure_inr_cr) and every function below runs unchanged against it — this is
the asset-level analogue of ingestion.financial_exposure.load_institution_portfolio.
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import DATA_DIR, HAZARD_WEIGHTS
from argus.common.provenance import DataProvenance, DataStatus
from argus.financial.impact_engine import HAZARD_COMPONENT_COLUMN
from argus.financial.vulnerability import HAZARDS, vulnerability_weight
from argus.ingestion.base import load_sample_csv

ASSET_LOCATIONS_PATH = DATA_DIR / "samples" / "asset_locations.csv"

REQUIRED_ASSET_COLUMNS = {"asset_id", "sector", "district", "lat", "lon", "asset_exposure_inr_cr"}


def load_demo_asset_locations() -> pd.DataFrame:
    """The bundled illustrative asset/borrower-location sample -- see module
    docstring for why this is explicitly demo data, never presented as real."""
    return load_sample_csv("asset_locations.csv")


def load_asset_locations(path: str | None = None) -> pd.DataFrame:
    """The swap-in point for a real institution's geolocated asset/borrower book.
    ``path=None`` (the default) loads the bundled illustrative demo file; a real
    deployment passes its own CSV with at least the REQUIRED_ASSET_COLUMNS."""
    if path is None:
        return load_demo_asset_locations()
    df = pd.read_csv(path)
    missing = REQUIRED_ASSET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Asset location file is missing required columns: {sorted(missing)}")
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = "false"  # a real institution file is, by construction, not synthetic
    return df


def compute_asset_level_exposure(assets: pd.DataFrame, hazard_scores: pd.DataFrame) -> pd.DataFrame:
    """assets: asset_id, asset_name (optional), sector, district, state (optional),
    lat, lon, asset_exposure_inr_cr. hazard_scores: the output of
    geospatial.hazard_scoring.compute_hazard_scores() (one row per district, with
    flood_component/drought_component/cyclone_component/heat_component).

    Returns one row per asset with its district's four hazard components carried
    down, the asset's own combined hazard x vulnerability sensitivity, and the
    resulting asset-level physical climate impact -- computed with the exact same
    ``HAZARD_WEIGHTS[hazard] x (score/100) x vulnerability_weight`` formula as
    financial/impact_engine.py's sector-level rollup (see that module's docstring
    for why folding in HAZARD_WEIGHTS this way keeps the result bounded), so an
    asset's impact is never inconsistent with its district/sector aggregate.
    """
    hz = hazard_scores.set_index("district")
    rows = []
    for _, a in assets.iterrows():
        district = a["district"]
        if district not in hz.index:
            continue
        hz_row = hz.loc[district]
        sector = a["sector"]
        exposure = float(a["asset_exposure_inr_cr"])

        combined_sensitivity = 0.0
        component_values = {}
        for hazard in HAZARDS:
            score = float(hz_row[HAZARD_COMPONENT_COLUMN[hazard]])
            weight = vulnerability_weight(sector, hazard)
            sensitivity_share = HAZARD_WEIGHTS[hazard] * (score / 100.0) * weight
            combined_sensitivity += sensitivity_share
            component_values[hazard] = round(score, 2)

        primary_hazard = max(HAZARDS, key=lambda h: component_values[h] * vulnerability_weight(sector, h))
        physical_impact = combined_sensitivity * exposure

        rows.append(
            {
                "asset_id": a["asset_id"],
                "asset_name": a.get("asset_name", a["asset_id"]),
                "sector": sector,
                "district": district,
                "state": a.get("state", hz_row.get("state")),
                "lat": a["lat"],
                "lon": a["lon"],
                "flood_component": component_values["flood"],
                "drought_component": component_values["drought"],
                "cyclone_component": component_values["cyclone"],
                "heat_component": component_values["heat"],
                "asset_exposure_inr_cr": round(exposure, 2),
                "combined_sensitivity": round(combined_sensitivity, 4),
                "asset_physical_impact_inr_cr": round(physical_impact, 2),
                "primary_hazard": primary_hazard,
                "is_synthetic": a.get("is_synthetic", "false"),
            }
        )
    return pd.DataFrame(rows).sort_values("asset_physical_impact_inr_cr", ascending=False).reset_index(drop=True)


def asset_locations_provenance(is_demo: bool = True) -> DataProvenance:
    if is_demo:
        return DataProvenance(
            dataset="Illustrative demo asset/borrower locations",
            source_organization="Argus (synthetic demo data -- no real institution portfolio configured)",
            source_url="data/samples/asset_locations.csv",
            publication_date="n/a -- not a published dataset",
            observation_period="n/a",
            geographic_resolution="point (lat/lon, placed near real district centroids)",
            unit="INR crore (asset_exposure_inr_cr)",
            methodology=(
                "Hand-constructed illustrative points near each demo district's real centroid, sector-"
                "labeled to be representative of that district's economy. Real, individually-geolocated "
                "borrower data is confidential by nature and cannot be shipped in a public project -- see "
                "geospatial/asset_exposure.py's module docstring for the swap-in interface a deploying "
                "institution uses to replace this file with its own real book."
            ),
            status=DataStatus.PROXY,
            retrieved_at=DataProvenance.now_iso(),
        )
    return DataProvenance(
        dataset="Institution asset/borrower locations",
        source_organization="Deploying institution (configured via load_asset_locations(path=...))",
        source_url="institution-internal",
        publication_date="n/a",
        observation_period="n/a",
        geographic_resolution="point (asset/borrower lat/lon)",
        unit="INR crore",
        methodology="Institution-supplied file; not computed by Argus.",
        status=DataStatus.OBSERVED,
        retrieved_at=DataProvenance.now_iso(),
    )


if __name__ == "__main__":
    from argus.geospatial.hazard_scoring import compute_hazard_scores
    from argus.ingestion.base import load_sample_csv as _load

    districts = _load("district_hazard_inputs.csv")[["district", "state", "lat", "lon"]]
    hazard_scores, _ = compute_hazard_scores(districts)
    assets = load_demo_asset_locations()
    result = compute_asset_level_exposure(assets, hazard_scores)
    print(result.to_string(index=False))
