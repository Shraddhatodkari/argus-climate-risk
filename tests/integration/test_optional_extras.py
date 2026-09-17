"""Integration tests: optional heavy extras (semantic-rag, augmentation) fail with a
clear, actionable error when their packages aren't installed — never a bare
ImportError/traceback — since neither is part of the default install."""

import pytest


def test_semantic_retriever_gives_actionable_error_without_extra():
    from argus.rag.embeddings import SemanticRetriever

    with pytest.raises(RuntimeError, match="semantic-rag"):
        SemanticRetriever()


def test_diffusion_augment_gives_actionable_error_without_extra():
    from argus.augmentation.diffusion_augment import augment_tile_diffusion
    from argus.ingestion.copernicus import synthetic_tile

    green, nir = synthetic_tile(0.3, seed=1)
    with pytest.raises(RuntimeError, match="augmentation"):
        augment_tile_diffusion(green, nir)
