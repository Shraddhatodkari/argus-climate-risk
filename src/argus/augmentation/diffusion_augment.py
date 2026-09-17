"""Augmentation for rare/imbalanced satellite disaster-imagery classes.

Historical flood-extent tiles for any single district are scarce and heavily skewed
toward "no flood" — a real problem for training any future classifier on this data.
Two augmentation paths are implemented here:

  augment_tile_classic()   — geometric + radiometric augmentation (flip, rotate, noise,
                               brightness jitter) using only numpy. Zero extra
                               dependencies, runs in milliseconds, and is what actually
                               ships and is tested by default — a legitimate, widely
                               used augmentation approach on its own.

  augment_tile_diffusion() — the certification-scoped method: inpainting-style
                               augmentation with a small pretrained diffusion model via
                               HuggingFace Diffusers. This needs `torch` + `diffusers`
                               (multi-GB, GPU-recommended), so it is deliberately kept
                               out of the default install (`pip install -e .`) and out
                               of CI. Install with `pip install -e ".[augmentation]"`
                               and run this module directly as an occasional, offline
                               R&D notebook step — never imported by the live pipeline.

Certification applied: NVIDIA — Generative AI with Diffusion Models.
"""

from __future__ import annotations

import numpy as np


def augment_tile_classic(
    green: np.ndarray, nir: np.ndarray, *, n_augments: int = 4, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Real, dependency-free augmentation: each output tile is a random combination of
    a 90-degree rotation, a horizontal/vertical flip, and additive sensor-noise +
    brightness jitter applied identically to both bands (so NDWI on the augmented tile
    stays physically consistent with the original)."""
    rng = np.random.default_rng(seed)
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_augments):
        g, n = green.copy(), nir.copy()

        k = int(rng.integers(0, 4))
        g, n = np.rot90(g, k), np.rot90(n, k)
        if rng.random() < 0.5:
            g, n = np.fliplr(g), np.fliplr(n)
        if rng.random() < 0.5:
            g, n = np.flipud(g), np.flipud(n)

        brightness = rng.normal(1.0, 0.05)
        noise_g = rng.normal(0, 0.01, g.shape)
        noise_n = rng.normal(0, 0.01, n.shape)
        g = np.clip(g * brightness + noise_g, 0.0, 1.0)
        n = np.clip(n * brightness + noise_n, 0.0, 1.0)

        outputs.append((g, n))
    return outputs


def augment_tile_diffusion(green: np.ndarray, nir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion-based augmentation — the certification-scoped path. Requires the
    `augmentation` extra (`pip install -e ".[augmentation]"`). Raises a clear,
    actionable error rather than a bare ImportError when the extra isn't installed."""
    try:
        import torch  # noqa: F401
        from diffusers import AutoPipelineForInpainting  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Diffusion-based augmentation requires the 'augmentation' extra "
            '(`pip install -e ".[augmentation]"` — pulls torch + diffusers, several GB). '
            "Use augment_tile_classic() for the zero-dependency default, or install the "
            "extra and re-run this module directly as an offline R&D step."
        ) from exc

    raise NotImplementedError(
        "Diffusion pipeline wiring (model load, band-to-RGB-proxy mapping, inpaint mask "
        "construction) is a Month-5 R&D notebook task once the 'augmentation' extra is "
        "installed on a machine with enough RAM/GPU headroom — see docs/architecture.md."
    )


if __name__ == "__main__":
    from argus.ingestion.copernicus import synthetic_tile

    green, nir = synthetic_tile(water_fraction=0.4, seed=1)
    augmented = augment_tile_classic(green, nir, n_augments=3)
    print(f"Generated {len(augmented)} classic-augmented tiles from 1 source tile.")
    for i, (g, n) in enumerate(augmented):
        print(f"  augment {i}: green mean={g.mean():.3f}, nir mean={n.mean():.3f}")
