"""Data-quality tests: every ingested dataset must pass its Pandera schema, and the
Data Agent must be the thing that enforces this before scoring ever runs."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from argus.agents.data_agent import run as data_agent_run
from argus.common.schemas import (
    DistrictHazardInputSchema,
    ExposureSchema,
    HazardScoreSchema,
    SatelliteWaterFractionSchema,
)
from argus.ingestion.base import load_sample_csv


def test_district_hazard_inputs_pass_schema():
    df = load_sample_csv("district_hazard_inputs.csv")
    DistrictHazardInputSchema.validate(df, lazy=True)  # raises on failure


def test_satellite_water_fraction_passes_schema():
    df = load_sample_csv("satellite_water_fraction.csv")
    SatelliteWaterFractionSchema.validate(df, lazy=True)


def test_data_agent_returns_validated_datasets():
    datasets = data_agent_run()
    assert "district_hazard_inputs" in datasets
    assert "satellite_water_fraction" in datasets
    assert len(datasets["district_hazard_inputs"]) > 0


def test_out_of_range_latitude_is_rejected():
    bad = pd.DataFrame(
        [{
            "district": "Test", "state": "Test", "lat": 200.0,  # out of -90..90 range — must fail
            "lon": 80.0, "source": "test", "is_proxy": "false",
        }]
    )
    with pytest.raises(SchemaErrors):
        DistrictHazardInputSchema.validate(bad, lazy=True)


def test_unrecognized_is_proxy_value_is_rejected():
    bad = pd.DataFrame(
        [{
            "district": "Test", "state": "Test", "lat": 20.0, "lon": 80.0,
            "source": "test", "is_proxy": "maybe",  # must be "true"/"false" — must fail
        }]
    )
    with pytest.raises(SchemaErrors):
        DistrictHazardInputSchema.validate(bad, lazy=True)


def test_hazard_score_schema_rejects_out_of_range_composite():
    bad = pd.DataFrame(
        [{
            "district": "Test", "flood_component": 50.0, "drought_component": 50.0,
            "cyclone_component": 50.0, "heat_component": 50.0,
            "composite_hazard_score": 150.0,  # out of 0-100 range — must fail
            "is_high_risk": True,
        }]
    )
    with pytest.raises(SchemaErrors):
        HazardScoreSchema.validate(bad, lazy=True)


def test_exposure_schema_rejects_negative_exposure():
    bad = pd.DataFrame(
        [{
            "district": "Test", "composite_hazard_score": 50.0,
            "public_exposure_inr_cr": -100.0,  # negative exposure — must fail
            "climate_exposed_exposure_inr_cr": 10.0, "is_high_risk": False,
        }]
    )
    with pytest.raises(SchemaErrors):
        ExposureSchema.validate(bad, lazy=True)
