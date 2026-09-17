"""Pandera schemas — the data-quality gate every ingested dataset must pass
before it reaches the scoring engine or the RAG index.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

# The district registry: district identity + centroid coordinates only. Every
# hazard/exposure figure that used to live in this file (flood_risk_wri,
# drought_risk, cyclone_risk, rainfall_anomaly_pct, priority_sector_credit_inr_cr)
# is now Derived at run time from real sources (ingestion/cyclone_imd.py,
# rainfall_imd.py, heat_imd.py, financial_exposure.py) rather than read from a
# static fixture — see docs/data-sources.md.
DistrictHazardInputSchema = DataFrameSchema(
    {
        "district": Column(str, Check.str_length(min_value=1)),
        "state": Column(str, Check.str_length(min_value=1)),
        "lat": Column(float, Check.in_range(-90.0, 90.0)),
        "lon": Column(float, Check.in_range(-180.0, 180.0)),
        "source": Column(str),
        "is_proxy": Column(str, Check.isin(["true", "false"])),
    },
    strict=False,
    coerce=True,
)

SatelliteWaterFractionSchema = DataFrameSchema(
    {
        "district": Column(str, Check.str_length(min_value=1)),
        "true_water_fraction": Column(float, Check.in_range(0.0, 1.0), coerce=True),
        "tile_note": Column(str),
        "source": Column(str),
        "is_synthetic": Column(str, Check.isin(["true", "false"])),
    },
    strict=False,
    coerce=True,
)

HazardScoreSchema = DataFrameSchema(
    {
        "district": Column(str),
        "flood_component": Column(float, Check.in_range(0, 100)),
        "drought_component": Column(float, Check.in_range(0, 100)),
        "cyclone_component": Column(float, Check.in_range(0, 100)),
        "heat_component": Column(float, Check.in_range(0, 100)),
        "composite_hazard_score": Column(float, Check.in_range(0, 100)),
        "is_high_risk": Column(bool),
    },
    strict=False,
    coerce=True,
)

ExposureSchema = DataFrameSchema(
    {
        "district": Column(str),
        "composite_hazard_score": Column(float, Check.in_range(0, 100)),
        "public_exposure_inr_cr": Column(float, Check.ge(0)),
        "climate_exposed_exposure_inr_cr": Column(float, Check.ge(0)),
        "is_high_risk": Column(bool),
    },
    strict=False,
    coerce=True,
)
