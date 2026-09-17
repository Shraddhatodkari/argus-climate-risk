"""Unit tests: asset/location-level geospatial intelligence (item #3 of the
Enterprise spec) — Borrower/asset -> lat/lon -> district -> hazard -> sector
vulnerability -> asset-level physical impact."""

import pandas as pd

from argus.geospatial.asset_exposure import (
    REQUIRED_ASSET_COLUMNS,
    compute_asset_level_exposure,
    load_asset_locations,
    load_demo_asset_locations,
)
from argus.geospatial.hazard_scoring import compute_hazard_scores


def test_demo_asset_locations_cover_every_demo_district(demo_districts):
    assets = load_demo_asset_locations()
    assert REQUIRED_ASSET_COLUMNS.issubset(assets.columns)
    assert set(assets["district"]) == set(demo_districts["district"])
    # Explicitly disclosed as illustrative, never silently presented as real.
    assert (assets["is_synthetic"] == "true").all()


def test_asset_level_impact_is_bounded_by_asset_exposure(demo_districts):
    hazard_scores, _ = compute_hazard_scores(demo_districts)
    assets = load_demo_asset_locations()
    result = compute_asset_level_exposure(assets, hazard_scores)
    assert not result.empty
    assert (result["asset_physical_impact_inr_cr"] <= result["asset_exposure_inr_cr"] + 0.01).all()
    assert (result["combined_sensitivity"] >= 0).all()
    assert (result["combined_sensitivity"] <= 1.0 + 1e-9).all()


def test_every_asset_carries_all_four_hazard_components(demo_districts):
    hazard_scores, _ = compute_hazard_scores(demo_districts)
    assets = load_demo_asset_locations()
    result = compute_asset_level_exposure(assets, hazard_scores)
    for col in ("flood_component", "drought_component", "cyclone_component", "heat_component"):
        assert col in result.columns
        assert result[col].notna().all()


def test_load_asset_locations_rejects_a_file_missing_required_columns(tmp_path):
    bad = tmp_path / "bad_assets.csv"
    pd.DataFrame([{"asset_id": "X-1", "district": "Cuttack"}]).to_csv(bad, index=False)
    try:
        load_asset_locations(str(bad))
        assert False, "should have raised on a file missing required asset columns"
    except ValueError:
        pass


def test_load_asset_locations_marks_a_real_institution_file_as_not_synthetic(tmp_path):
    real = tmp_path / "real_assets.csv"
    pd.DataFrame(
        [{"asset_id": "X-1", "sector": "MSME", "district": "Cuttack", "lat": 20.46, "lon": 85.88, "asset_exposure_inr_cr": 10.0}]
    ).to_csv(real, index=False)
    df = load_asset_locations(str(real))
    assert (df["is_synthetic"] == "false").all()
