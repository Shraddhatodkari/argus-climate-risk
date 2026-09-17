"""Shared pytest fixtures."""

import pandas as pd
import pytest

from argus.ingestion.base import load_sample_csv


@pytest.fixture
def demo_districts() -> pd.DataFrame:
    """The 8-district registry (district/state/lat/lon) every hazard/exposure
    computation in this Enterprise build takes as input."""
    return load_sample_csv("district_hazard_inputs.csv")[["district", "state", "lat", "lon"]]
