"""Sector allocation — splits a district's public financial exposure estimate
(ingestion/financial_exposure.py) into Agriculture / MSME / Other, so the
vulnerability model (financial/vulnerability.py) has something sector-specific to
apply its sensitivity weights to.

Uses a real, state-specific NABARD sector split where this build actually has one
on file (Maharashtra, Uttar Pradesh — see data/real/nabard_sector_composition.csv,
each row cited and dated), and falls back to a real but non-state-specific NABARD
sector composition (52% Agriculture / 39% MSME / 9% Other, also cited and dated in
the same CSV) for every other state in this demo set. This is disclosed explicitly
via each row's ``split_status`` field — DERIVED (Observed) for a state with its own
cited split, DERIVED (Proxy) where the representative national split stands in for
a state-specific one this build does not have on file. Never silently uniform.
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import DATA_DIR, DEFAULT_SECTOR_SPLIT
from argus.common.provenance import DataProvenance, DataStatus

SECTOR_COMPOSITION_PATH = DATA_DIR / "real" / "nabard_sector_composition.csv"
SECTORS = ["Agriculture", "MSME", "Other"]


def _load_sector_splits() -> dict[str, dict[str, float]]:
    df = pd.read_csv(SECTOR_COMPOSITION_PATH)
    splits = {}
    for _, row in df[df["scope"] != "representative_state_2024"].iterrows():
        splits[row["scope"]] = {
            "Agriculture": float(row["agriculture_share"]),
            "MSME": float(row["msme_share"]),
            "Other": float(row["other_share"]),
        }
    return splits


def allocate_sectors(district_exposure: pd.DataFrame) -> tuple[pd.DataFrame, DataProvenance]:
    """district_exposure needs 'district', 'state', 'public_exposure_inr_cr'.
    Returns one row per (district, sector): district, state, sector,
    sector_exposure_inr_cr, sector_share, split_status."""
    retrieved_at = DataProvenance.now_iso()
    state_splits = _load_sector_splits()

    rows = []
    for _, d in district_exposure.iterrows():
        state = d["state"]
        split = state_splits.get(state, DEFAULT_SECTOR_SPLIT)
        split_status = DataStatus.DERIVED if state in state_splits else DataStatus.PROXY
        for sector in SECTORS:
            share = split[sector]
            rows.append(
                {
                    "district": d["district"],
                    "state": state,
                    "sector": sector,
                    "sector_share": share,
                    "sector_exposure_inr_cr": round(float(d["public_exposure_inr_cr"]) * share, 1),
                    "split_status": split_status.value,
                }
            )

    provenance = DataProvenance(
        dataset="Sector allocation of district public exposure (NABARD sector composition)",
        source_organization="NABARD",
        source_url="see data/real/nabard_sector_composition.csv",
        publication_date="mixed -- state-specific rows dated per state; representative split dated 2024-03-05",
        observation_period="FY2024-25 to FY2026-27",
        geographic_resolution="state (sector split) x district (exposure base)",
        unit="INR crore",
        methodology="sector_exposure = district public_exposure_inr_cr x sector_share (state-specific NABARD split where on file, else the representative national split).",
        status=DataStatus.DERIVED,
        retrieved_at=retrieved_at,
    )
    return pd.DataFrame(rows), provenance


if __name__ == "__main__":
    demo = pd.DataFrame(
        [
            {"district": "Latur", "state": "Maharashtra", "public_exposure_inr_cr": 22455.0},
            {"district": "Puri", "state": "Odisha", "public_exposure_inr_cr": 12748.0},
        ]
    )
    result, prov = allocate_sectors(demo)
    print(result.to_string(index=False))
