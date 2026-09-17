"""Satellite flood-extent detection via NDWI (Normalized Difference Water Index).

NDWI = (Green - NIR) / (Green + NIR). Open water reflects green light and strongly
absorbs near-infrared, giving NDWI values that are positive and typically well above
vegetation/soil. This is the standard, well-established lightweight remote-sensing
technique for flood-extent mapping — deliberately chosen over a trained CNN/U-Net
segmentation model so it runs on a CPU-only laptop with no training data and no GPU;
docs/architecture.md documents the pretrained-segmentation upgrade path for later.

Certification applied: NVIDIA — Disaster Risk Monitoring Using Satellite Imagery.
"""

from __future__ import annotations

import numpy as np

NDWI_WATER_THRESHOLD = 0.0


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Elementwise NDWI. Adds a small epsilon to avoid division by zero on a
    fully-dark pixel (a real, if rare, satellite artifact)."""
    green = green.astype(np.float64)
    nir = nir.astype(np.float64)
    return (green - nir) / (green + nir + 1e-9)


def flood_extent_fraction(green: np.ndarray, nir: np.ndarray, *, threshold: float = NDWI_WATER_THRESHOLD) -> float:
    """Fraction of pixels classified as open water — the district's satellite-derived
    flood-extent signal, on a 0.0-1.0 scale."""
    ndwi = compute_ndwi(green, nir)
    return float(np.mean(ndwi > threshold))


def flood_extent_score(green: np.ndarray, nir: np.ndarray, *, threshold: float = NDWI_WATER_THRESHOLD) -> float:
    """0-100 scaled version of flood_extent_fraction, for direct use in the composite
    hazard score."""
    return flood_extent_fraction(green, nir, threshold=threshold) * 100.0


if __name__ == "__main__":
    from argus.ingestion.copernicus import load_district_tiles

    for district, (green, nir) in load_district_tiles().items():
        print(f"{district:15s} flood_extent_score={flood_extent_score(green, nir):.1f}")
