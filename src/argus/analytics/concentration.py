"""Concentration analysis -- rolls the district-sector-hazard impact detail up
along State -> District -> Sector -> Hazard -> Exposure, and turns the largest
concentration into a plain-English, management-readable insight sentence, rather
than leaving a risk officer to eyeball a sorted table and draw that conclusion
themselves.

This is deliberately simple, deterministic aggregation (groupby + sort + a small
templated sentence) -- no LLM involved in deciding WHICH concentration is largest;
an LLM is only ever used downstream (if at all) to restate an already-computed
finding in different words, consistent with this project's "deterministic engine
computes, LLM explains" boundary (see docs/architecture.md).
"""

from __future__ import annotations

import pandas as pd


def concentration_by_state(impact_detail: pd.DataFrame) -> pd.DataFrame:
    if impact_detail.empty:
        return impact_detail
    return (
        impact_detail.groupby("state", as_index=False)["physical_impact_inr_cr"]
        .sum()
        .sort_values("physical_impact_inr_cr", ascending=False)
        .reset_index(drop=True)
    )


def concentration_by_hazard(impact_detail: pd.DataFrame) -> pd.DataFrame:
    if impact_detail.empty:
        return impact_detail
    return (
        impact_detail.groupby("hazard", as_index=False)["physical_impact_inr_cr"]
        .sum()
        .sort_values("physical_impact_inr_cr", ascending=False)
        .reset_index(drop=True)
    )


def concentration_by_sector(impact_detail: pd.DataFrame) -> pd.DataFrame:
    if impact_detail.empty:
        return impact_detail
    return (
        impact_detail.groupby("sector", as_index=False)["physical_impact_inr_cr"]
        .sum()
        .sort_values("physical_impact_inr_cr", ascending=False)
        .reset_index(drop=True)
    )


def concentration_by_state_hazard(impact_detail: pd.DataFrame) -> pd.DataFrame:
    """The cross-cut that produces the user-facing insight: which state x hazard
    pair carries the largest concentration of exposed lending."""
    if impact_detail.empty:
        return impact_detail
    return (
        impact_detail.groupby(["state", "hazard"], as_index=False)["physical_impact_inr_cr"]
        .sum()
        .sort_values("physical_impact_inr_cr", ascending=False)
        .reset_index(drop=True)
    )


def top_concentration_insight(impact_detail: pd.DataFrame) -> str:
    """One templated sentence naming the single largest state x hazard
    concentration and what share of total physical impact it represents --
    e.g. "38% of total physical climate impact is concentrated in Odisha's
    exposure to cyclone hazard.\""""
    by_state_hazard = concentration_by_state_hazard(impact_detail)
    if by_state_hazard.empty:
        return "No physical impact data available to assess concentration."
    top = by_state_hazard.iloc[0]
    total = impact_detail["physical_impact_inr_cr"].sum()
    share_pct = 100.0 * top["physical_impact_inr_cr"] / total if total else 0.0
    return (
        f"{share_pct:.0f}% of total physical climate impact (Rs {top['physical_impact_inr_cr']:,.0f} Cr of "
        f"Rs {total:,.0f} Cr) is concentrated in {top['state']}'s exposure to {top['hazard']} hazard."
    )


def sector_geography_insight(impact_detail: pd.DataFrame) -> str:
    """A second insight sentence pairing the top sector with its top state, in the
    style of the example the spec gives: "A large share of exposed lending is
    concentrated in <sector> in <state>, which carries elevated <hazard> exposure.\""""
    if impact_detail.empty:
        return "No physical impact data available to assess sector concentration."
    by_sector_state = (
        impact_detail.groupby(["sector", "state"], as_index=False)["physical_impact_inr_cr"].sum().sort_values("physical_impact_inr_cr", ascending=False)
    )
    top = by_sector_state.iloc[0]
    top_hazard_row = (
        impact_detail[(impact_detail["sector"] == top["sector"]) & (impact_detail["state"] == top["state"])]
        .groupby("hazard")["physical_impact_inr_cr"]
        .sum()
        .idxmax()
    )
    return (
        f"A large share of exposed lending in the {top['sector']} sector is concentrated in {top['state']}, "
        f"which carries elevated {top_hazard_row} exposure."
    )


if __name__ == "__main__":
    from argus.geospatial.exposure_engine import run_pipeline
    from argus.ingestion.base import load_sample_csv

    demo = load_sample_csv("district_hazard_inputs.csv")[["district", "state", "lat", "lon"]]
    output = run_pipeline(demo)
    print(top_concentration_insight(output.impact_detail))
    print(sector_geography_insight(output.impact_detail))
    print(concentration_by_state_hazard(output.impact_detail).to_string(index=False))
