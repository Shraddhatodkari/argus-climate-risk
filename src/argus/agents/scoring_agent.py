"""Hazard & Exposure Agent — runs the composite hazard score and Climate-Adjusted
Exposure at Risk computation for every district. Thin wrapper over
geospatial/exposure_engine.py; exists as its own agent (rather than being inlined into
the orchestrator) so it has its own testable boundary and, in the MCP layer, its own
callable tool.
"""

from __future__ import annotations

import pandas as pd

from argus.agents.data_agent import run as data_agent_run
from argus.geospatial.exposure_engine import PipelineOutput, compute_exposure_at_risk, run_pipeline


def _districts() -> pd.DataFrame:
    datasets = data_agent_run()
    return datasets["district_hazard_inputs"][["district", "state", "lat", "lon"]]


def run() -> pd.DataFrame:
    return compute_exposure_at_risk(_districts())


def run_full_pipeline() -> PipelineOutput:
    """Returns every intermediate table (hazard scores, sector exposure, impact
    detail, provenance) -- what the Streamlit app's deeper tabs and the Data
    Lineage page need, beyond the flat exposure table ``run()`` returns."""
    return run_pipeline(_districts())


def run_for_district(district: str) -> dict:
    df = run()
    match = df[df["district"].str.lower() == district.lower()]
    if match.empty:
        raise ValueError(f"No hazard/exposure data for district {district!r}.")
    return match.iloc[0].to_dict()


if __name__ == "__main__":
    print(run().to_string(index=False))
