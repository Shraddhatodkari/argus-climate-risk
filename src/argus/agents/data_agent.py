"""Data Agent — refreshes and validates every raw source before anything downstream
touches it. Nothing else in the pipeline is allowed to read a source file directly;
everything goes through here first so a schema failure stops the pipeline at the
earliest possible point, not three agents later.

``district_hazard_inputs.csv`` is, despite its legacy filename, now just the
district REGISTRY (district/state/lat/lon) -- every hazard and exposure figure that
used to live in it is Derived at run time by the real-data ingestion modules
(cyclone_imd, rainfall_imd, heat_imd, financial_exposure) instead. See
common/schemas.py's DistrictHazardInputSchema docstring.
"""

from __future__ import annotations

import pandas as pd

from argus.common.schemas import DistrictHazardInputSchema, SatelliteWaterFractionSchema
from argus.ingestion.base import load_sample_csv


class DataValidationError(Exception):
    pass


def run() -> dict[str, pd.DataFrame]:
    """Loads and schema-validates every raw dataset the pipeline depends on. Raises
    DataValidationError with the pandera failure detail on any violation, rather than
    letting a malformed row silently propagate into a risk score."""
    hazard_inputs = load_sample_csv("district_hazard_inputs.csv")
    satellite = load_sample_csv("satellite_water_fraction.csv")

    try:
        DistrictHazardInputSchema.validate(hazard_inputs, lazy=True)
    except Exception as exc:  # pandera.errors.SchemaErrors
        raise DataValidationError(f"district_hazard_inputs.csv failed validation: {exc}") from exc

    try:
        SatelliteWaterFractionSchema.validate(satellite, lazy=True)
    except Exception as exc:
        raise DataValidationError(f"satellite_water_fraction.csv failed validation: {exc}") from exc

    return {"district_hazard_inputs": hazard_inputs, "satellite_water_fraction": satellite}


if __name__ == "__main__":
    datasets = run()
    for name, df in datasets.items():
        print(f"{name}: {len(df)} rows, schema OK")
