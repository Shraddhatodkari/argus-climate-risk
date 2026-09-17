"""Shared ingestion helpers.

Every ingestion module follows the same real-world-honest pattern: attempt a live
HTTP fetch from the documented public endpoint first; if the network is unavailable
or the response shape doesn't match (common in a sandboxed/offline environment, and a
real concern in any restricted-egress enterprise network), fall back to the bundled
sample dataset in data/samples/ — clearly flagged is_proxy / is_synthetic in every row
— so the pipeline is always runnable, testable, and demoable without a live connection.

This mirrors how a real institution would actually operate: a resilient pipeline that
degrades to its last-known-good snapshot rather than failing hard when one upstream
source is unreachable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from argus.common.config import SAMPLES_DIR

logger = logging.getLogger("argus.ingestion")


def safe_get(url: str, *, timeout: float = 8.0, params: dict | None = None) -> requests.Response | None:
    """GET a URL, returning None (never raising) on any network failure. Ingestion
    modules must never let a flaky upstream source crash the pipeline."""
    try:
        resp = requests.get(url, timeout=timeout, params=params)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("Live fetch failed for %s (%s) — falling back to sample data.", url, exc)
        return None


def load_sample_csv(filename: str) -> pd.DataFrame:
    path: Path = SAMPLES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Sample fixture {path} is missing. Run `python -m argus.ingestion.<module>` "
            "or restore data/samples/ from the repository."
        )
    df = pd.read_csv(path)
    # pandas' CSV parser auto-infers lowercase true/false literals as bool — but every
    # flag column here (is_proxy / is_synthetic) is deliberately kept as the literal
    # string "true"/"false" downstream (schemas, equality checks, audit records), so
    # normalize it back explicitly rather than let type inference silently vary it.
    for col in df.columns:
        if col.startswith("is_"):
            df[col] = df[col].astype(str).str.lower()
    return df
