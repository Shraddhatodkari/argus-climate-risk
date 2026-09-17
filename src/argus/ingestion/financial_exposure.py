"""Real public financial exposure — replaces the old illustrative
₹4,200-5,100cr fixture with a genuinely sourced, two-layer estimate, and is
explicit everywhere about what it is and is not:

    Public financial exposure estimate   -- what this module computes
             vs.
    Institution portfolio exposure       -- a real bank/NBFC's actual book,
                                             NOT present in this build; see
                                             ``load_institution_portfolio()``
                                             below for the swap-in interface.

Methodology (Derived, from two real Observed sources):
1. State-level priority-sector credit POTENTIAL, as actually announced by NABARD at
   each state's annual State Credit Seminar / State Focus Paper release — a real,
   dated, publicly reported figure for every state in this demo (see
   data/real/nabard_state_credit_potential.csv for the exact figure, fiscal year,
   publication date and source URL cited per state; six separate announcements,
   deliberately not all the same fiscal year, because that is what NABARD actually
   published for each state as of this build's retrieval date -- each one is
   individually dated rather than forced into a false single-vintage table).
2. Real Census 2011 district and state population
   (data/real/census2011_population.csv, sourced per row).
3. District estimate = state total x (district population / state population) --
   a standard, transparent apportionment. It is NOT a claim that NABARD publishes
   this number at district granularity (it mostly doesn't, publicly, at the
   consistent scale this project needs) -- it is a Derived estimate of a district's
   share of its state's real, disclosed priority-sector credit potential, and every
   value returned says so via its DataProvenance.status = DERIVED.

What this number means: NABARD's "credit potential" figures are the addressable
priority-sector credit market for the whole geography -- i.e. how much priority-
sector lending the local economy could absorb -- not a single institution's
outstanding loan book. That is a real and useful number for climate-risk exposure
screening (it is what a lender's *addressable* geography-linked exposure looks
like), but it is a different concept from "our bank's book in this district," and
the dashboard must never blur the two. Hence the explicit institution-portfolio
hook below.
"""

from __future__ import annotations

import pandas as pd

from argus.common.config import DATA_DIR
from argus.common.provenance import DataProvenance, DataStatus

NABARD_CREDIT_PATH = DATA_DIR / "real" / "nabard_state_credit_potential.csv"
CENSUS_PATH = DATA_DIR / "real" / "census2011_population.csv"


def load_public_exposure(districts: pd.DataFrame) -> tuple[pd.DataFrame, DataProvenance]:
    """districts needs columns 'district', 'state'. Returns district, state,
    public_exposure_inr_cr, state_credit_potential_inr_cr, state_fiscal_year,
    district_population_2011, state_population_2011, district_population_share --
    every intermediate value a Data Lineage view would show as the calculation's
    working."""
    retrieved_at = DataProvenance.now_iso()
    credit = pd.read_csv(NABARD_CREDIT_PATH)
    census = pd.read_csv(CENSUS_PATH)

    state_pop = census[census["level"] == "state"].set_index("state")["population_2011"]
    district_pop = census[census["level"] == "district"].set_index("name")["population_2011"]

    rows = []
    for _, d in districts.iterrows():
        district, state = d["district"], d["state"]
        credit_row = credit[credit["state"] == state]
        if credit_row.empty or district not in district_pop.index or state not in state_pop.index:
            continue
        credit_row = credit_row.iloc[0]
        d_pop = int(district_pop.loc[district])
        s_pop = int(state_pop.loc[state])
        share = d_pop / s_pop
        estimate = float(credit_row["credit_potential_inr_cr"]) * share
        rows.append(
            {
                "district": district,
                "state": state,
                "public_exposure_inr_cr": round(estimate, 1),
                "state_credit_potential_inr_cr": float(credit_row["credit_potential_inr_cr"]),
                "state_fiscal_year": credit_row["fiscal_year"],
                "district_population_2011": d_pop,
                "state_population_2011": s_pop,
                "district_population_share": round(share, 5),
            }
        )
    result = pd.DataFrame(rows)

    provenance = DataProvenance(
        dataset="Public financial exposure estimate (NABARD state credit potential x Census 2011 district population share)",
        source_organization="NABARD (state credit potential) + Registrar General & Census Commissioner of India (population)",
        source_url="see data/real/nabard_state_credit_potential.csv and data/real/census2011_population.csv for the exact URL cited per state/district",
        publication_date="mixed by state -- see state_fiscal_year and each source's own publication_date in the CSVs above",
        observation_period="Census 2011 (population); FY2025-26 to FY2026-27 (state credit potential, per state)",
        geographic_resolution="district (Derived by population-share apportionment of a real state total)",
        unit="INR crore",
        methodology=(
            "district_estimate = state_credit_potential x (district_population_2011 / state_population_2011). "
            "See financial_exposure.py's module docstring for the full reasoning and its limits."
        ),
        status=DataStatus.DERIVED,
        retrieved_at=retrieved_at,
    )
    return result, provenance


def load_institution_portfolio(path: str | None = None) -> pd.DataFrame | None:
    """The swap-in point for a real bank/NBFC's confidential portfolio. Expects a CSV
    at ``path`` with at minimum: district, sector, exposure_inr_cr (asset/borrower-
    level lat/lon and sector detail are supported -- see geospatial/asset_exposure.py).
    Returns None (not an empty DataFrame) when no institution file is configured, so
    every caller must explicitly branch on "no institution portfolio in this build"
    rather than silently treating an empty table as zero exposure.

    No institution portfolio ships with this project -- doing so would mean either
    fabricating a fake bank's book (which this project's whole design philosophy
    refuses to do) or exposing a real one. This function exists so a deploying
    institution has exactly one place to plug in their real book; every module that
    computes exposure/impact accepts an optional institution_portfolio and falls back
    to the public estimate above (clearly labeled) when it is not supplied.
    """
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"district", "sector", "exposure_inr_cr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Institution portfolio file is missing required columns: {sorted(missing)}")
    return df


if __name__ == "__main__":
    demo = pd.DataFrame(
        [
            {"district": "Kanchipuram", "state": "Tamil Nadu"},
            {"district": "Kamrup", "state": "Assam"},
        ]
    )
    result, prov = load_public_exposure(demo)
    print(result.to_string(index=False))
    print(prov)
