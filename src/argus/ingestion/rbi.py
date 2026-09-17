"""RBI DBIE ingestion — macro-financial context series (Handbook of Statistics).

Used by the Narrative Agent for macro framing, not by the district hazard/exposure
engine. Live source: RBI's Database on Indian Economy (DBIE), api.rbi.org.in / public
CSV export. Falls back to a small embedded macro snapshot when unreachable, clearly
flagged as illustrative.
"""

from __future__ import annotations

import logging

import pandas as pd

from argus.ingestion.base import safe_get

DBIE_URL = "https://api.rbi.org.in/dbie/series"

_LOGGER = logging.getLogger(__name__)

_FALLBACK_MACRO_SNAPSHOT = pd.DataFrame(
    [
        {"series": "priority_sector_lending_growth_yoy_pct", "value": 11.4, "is_proxy": "true"},
        {"series": "scheduled_commercial_bank_gnpa_pct", "value": 2.6, "is_proxy": "true"},
    ]
)


def load_macro_context() -> pd.DataFrame:
    resp = safe_get(DBIE_URL)
    if resp is not None:
        try:
            live_df = pd.DataFrame(resp.json())
            if {"series", "value"}.issubset(live_df.columns):
                live_df["is_proxy"] = "false"
                return live_df
        except Exception:
            _LOGGER.debug("RBI DBIE live fetch returned an unparseable payload; using fallback snapshot.", exc_info=True)
    return _FALLBACK_MACRO_SNAPSHOT.copy()


if __name__ == "__main__":
    print(load_macro_context().to_string(index=False))
