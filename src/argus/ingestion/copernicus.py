"""Copernicus Emergency Management Service (EMS) ingestion — satellite disaster extent.

Live source: Copernicus EMS rapid-mapping activations (public, free) or Sentinel-2
Level-2A green/NIR reflectance tiles. This sandbox cannot reach Copernicus's servers,
so this module deterministically generates a small synthetic green/NIR reflectance
tile per district from the bundled ``true_water_fraction`` label in
data/samples/satellite_water_fraction.csv — enough surface area (real band arrays, a
real NDWI computation in geospatial/flood_index.py, a known ground truth) to build and
unit-test the actual satellite hazard-detection pipeline without a live imagery feed.

Swap ``fetch_live_tile()`` for a real Sentinel-2 / Copernicus EMS download in
production; nothing downstream changes, since both paths return the same
(green_band, nir_band) numpy array pair.

Certification applied: NVIDIA — Disaster Risk Monitoring Using Satellite Imagery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argus.ingestion.base import load_sample_csv


def fetch_live_tile(district: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Real integration point for a live Copernicus/Sentinel-2 pull. Returns None in
    this environment (no reachable imagery API); a production build replaces this
    function body with an authenticated download + band read via rasterio."""
    return None


def synthetic_tile(water_fraction: float, *, size: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic green/NIR reflectance tile where approximately
    ``water_fraction`` of pixels carry water-like reflectance (NDWI-positive: green
    reflectance exceeds NIR, since open water absorbs near-infrared light) and the rest
    carry land-like reflectance (NIR exceeds green). Used as a stand-in for a real
    Sentinel-2 tile so the NDWI pipeline has real array math to run and a known ground
    truth to validate against.
    """
    rng = np.random.default_rng(seed)
    n_pixels = size * size
    is_water = rng.random(n_pixels) < water_fraction

    green = np.where(
        is_water,
        rng.normal(0.30, 0.03, n_pixels),   # water: moderate green reflectance
        rng.normal(0.12, 0.03, n_pixels),   # land: lower green reflectance
    )
    nir = np.where(
        is_water,
        rng.normal(0.05, 0.02, n_pixels),   # water: strongly absorbs NIR
        rng.normal(0.35, 0.05, n_pixels),   # land/vegetation: strong NIR reflectance
    )
    green = np.clip(green, 0.0, 1.0).reshape(size, size)
    nir = np.clip(nir, 0.0, 1.0).reshape(size, size)
    return green, nir


def load_district_tiles() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {district: (green_band, nir_band)} for every district in the sample
    satellite fixture, preferring a live tile where one is reachable."""
    df: pd.DataFrame = load_sample_csv("satellite_water_fraction.csv")
    tiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, row in df.iterrows():
        live = fetch_live_tile(row["district"])
        tiles[row["district"]] = live if live is not None else synthetic_tile(
            row["true_water_fraction"], seed=i
        )
    return tiles


if __name__ == "__main__":
    for district, (g, n) in load_district_tiles().items():
        print(district, g.shape, n.shape)
