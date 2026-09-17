"""Real cyclone hazard — IMD RSMC New Delhi best-track record via the ``imdtrack``
package (PyPI), with a frozen local snapshot as the offline fallback.

This replaces the illustrative ``cyclone_risk`` fixture column used in the V1 demo
build. There is no synthetic data anywhere in this module: every number is either a
live-fetched or a locally-snapshotted observation from IMD's own published record,
and the per-district exposure score is a disclosed, reproducible calculation over
those observations — nothing here is asserted without a citation back to the
best-track row(s) that produced it.

Source: India Meteorological Department, RSMC New Delhi — Tropical Cyclones over
North Indian Ocean, the official best-track archive (every depression and cyclonic
storm in the Bay of Bengal & Arabian Sea since 1982). Accessed via the ``imdtrack``
Python package (https://github.com/syedhamidali/imdtrack, MIT-licensed, PyPI:
imdtrack), which parses IMD's published workbook into a tidy table and refreshes it
monthly from IMD's own source.

Methodology — district cyclone exposure score (0-100):
1. For each district centroid, take every best-track observation (a 3-hourly storm
   fix) within ``PROXIMITY_RADIUS_KM`` of that point. This radius is a documented
   assumption (150 km), not a hazard boundary IMD itself publishes — it approximates
   the reach of damaging wind/storm-surge/rainfall impact around a track, consistent
   with typical impact radii used in cyclone risk literature, but a production
   deployment should replace it with a wind-field or storm-surge model if available.
2. Group by storm (a storm may pass near a district more than once); take that
   storm's MAX wind speed while within radius — the closest approach to peak local
   impact this data can represent without a full wind-field model.
3. Weight each storm by recency: a half-life of ``RECENCY_HALF_LIFE_YEARS`` years, so
   a storm 25 years old counts half as much as one from this year. This is a
   disclosed modelling choice (more recent storms are more informative about current
   risk, and monsoon/cyclone climatology does shift over decades) — not a hazard
   number invented from nothing; every storm that contributes is a real,
   IMD-recorded event.
4. Sum (weight x max_wind/100) across storms for a district's raw score, then map it
   to 0-100 via a saturating exponential (``1 - exp(-raw/SATURATION_K)``) so a
   handful of extreme historical landfalls (e.g. coastal Odisha) don't blow the
   scale out for everywhere else, while staying monotonic and reproducible.

This whole calculation is Derived (see common/provenance.py) from Observed IMD
data — status/provenance is returned alongside the score, not just the number.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from argus.common.config import DATA_DIR
from argus.common.provenance import DataProvenance, DataStatus

_LOGGER = logging.getLogger(__name__)

LOCAL_SNAPSHOT_PATH = DATA_DIR / "real" / "imd_cyclone_besttrack.csv"

PROXIMITY_RADIUS_KM = 150.0
RECENCY_HALF_LIFE_YEARS = 25.0
SATURATION_K = 6.0
REFERENCE_WIND_KT = 100.0  # normalisation reference, not a physical cap


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_best_track() -> tuple[pd.DataFrame, DataProvenance]:
    """Returns (observations, provenance). Tries a live fetch via ``imdtrack`` first
    (it re-downloads its pre-parsed table from GitHub only if the source changed);
    falls back to this build's frozen local snapshot if the package or network is
    unavailable, exactly like every other ingestion module in this project."""
    retrieved_at = DataProvenance.now_iso()
    try:
        import imdtrack as imd  # optional at runtime; declared in pyproject's "gis"-adjacent extras below

        bt = imd.load()
        df = bt.observations
        status = DataStatus.OBSERVED
        obs_period = f"{int(df['year'].min())}-{int(df['year'].max())}"
    except Exception:
        _LOGGER.debug("Live imdtrack fetch unavailable; using the frozen local snapshot.", exc_info=True)
        df = pd.read_csv(LOCAL_SNAPSHOT_PATH, parse_dates=["time"])
        status = DataStatus.OBSERVED  # still a real IMD record, just not re-fetched this run
        obs_period = f"{int(df['year'].min())}-{int(df['year'].max())}"

    provenance = DataProvenance(
        dataset="IMD RSMC New Delhi cyclone best-track record",
        source_organization="India Meteorological Department (RSMC New Delhi)",
        source_url="https://rsmcnewdelhi.imd.gov.in/",
        publication_date="continuously updated (monthly)",
        observation_period=obs_period,
        geographic_resolution="storm fix (lat/lon, ~3-hourly)",
        unit="knots (wind), hPa (pressure)",
        methodology=(
            "IMD's official hand-maintained best-track workbook, parsed into a tidy table by the "
            "imdtrack package (github.com/syedhamidali/imdtrack); district exposure is Derived from "
            "these Observed fixes — see cyclone_imd.py's module docstring for the exact formula."
        ),
        status=status,
        retrieved_at=retrieved_at,
    )
    return df[["storm_id", "year", "basin", "name", "time", "lat", "lon", "wind", "pressure", "grade"]], provenance


def compute_district_cyclone_exposure(districts: pd.DataFrame) -> tuple[pd.DataFrame, DataProvenance]:
    """districts must have columns district, lat, lon. Returns a DataFrame with
    district, cyclone_risk (0-100), storm_count, max_wind_kt_observed — every one of
    those last two columns is what a Data Lineage view would cite as the evidence
    behind the score."""
    observations, provenance = load_best_track()
    observations = observations.dropna(subset=["lat", "lon", "wind"])

    rows = []
    for _, d in districts.iterrows():
        dist_km = _haversine_km(d["lat"], d["lon"], observations["lat"].values, observations["lon"].values)
        nearby = observations[dist_km <= PROXIMITY_RADIUS_KM]
        if nearby.empty:
            rows.append({"district": d["district"], "cyclone_risk": 0.0, "storm_count": 0, "max_wind_kt_observed": 0.0})
            continue
        per_storm = nearby.groupby("storm_id").agg(max_wind=("wind", "max"), year=("year", "first"))
        current_year = pd.Timestamp.now("UTC").year
        weight = 0.5 ** ((current_year - per_storm["year"]) / RECENCY_HALF_LIFE_YEARS)
        raw_score = float((weight * per_storm["max_wind"] / REFERENCE_WIND_KT).sum())
        score = 100.0 * (1.0 - np.exp(-raw_score / SATURATION_K))
        rows.append(
            {
                "district": d["district"],
                "cyclone_risk": round(float(score), 2),
                "storm_count": len(per_storm),
                "max_wind_kt_observed": float(per_storm["max_wind"].max()),
            }
        )
    return pd.DataFrame(rows), provenance


if __name__ == "__main__":
    demo = pd.DataFrame(
        [
            {"district": "Puri", "lat": 19.8135, "lon": 85.8312},
            {"district": "Kamrup", "lat": 26.1445, "lon": 91.7362},
        ]
    )
    result, prov = compute_district_cyclone_exposure(demo)
    print(result)
    print(prov)
