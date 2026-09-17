"""Data provenance — the single record type every hazard and exposure dataset in
Argus Enterprise must carry, and the backbone of the Data Lineage page.

Design rationale: an enterprise climate-risk figure is only as trustworthy as its
audit trail. A composite score with no citation is a number a risk committee cannot
defend to a regulator or an auditor. Every dataset ingested by this pipeline —
whether fetched live, computed from a real historical record, or (during a network
outage) served from a frozen snapshot — carries one of these records, and every
number derived from it inherits that record's provenance rather than losing it.

``DataStatus`` is the four-way label the dashboard shows next to every value:

- OBSERVED — a direct measurement or government-published record (e.g. an IMD
  cyclone best-track fix, a published rainfall total).
- DERIVED  — computed from Observed data via a disclosed, deterministic formula
  (e.g. a district cyclone-exposure score computed from best-track proximity, or a
  district credit estimate allocated from a real state total via Census population
  share). Derived is not a euphemism for fabricated — the methodology is always
  documented and reproducible from Observed inputs.
- MODELLED — an output of a named external model or scenario framework (e.g. an
  NGFS transition/physical scenario multiplier) rather than a direct observation.
- PROXY    — a stand-in used because the real figure is unavailable in this build
  (e.g. a district-level number substituted with its state or subdivision average
  because no finer-grained public source could be found), always labeled and never
  presented as if it were Observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class DataStatus(str, Enum):
    OBSERVED = "Observed"
    DERIVED = "Derived"
    MODELLED = "Modelled"
    PROXY = "Proxy"


@dataclass(frozen=True)
class DataProvenance:
    """One record per dataset/field. Immutable, JSON-serializable (via asdict), and
    designed to be stored alongside every hazard/exposure value it backs."""

    dataset: str
    source_organization: str
    source_url: str
    publication_date: str  # ISO date the source organization published this record/version
    observation_period: str  # e.g. "1982-2026", "2011" (a single-year census), "FY2024-25"
    geographic_resolution: str  # e.g. "district centroid", "IMD meteorological subdivision (36 units)", "state"
    unit: str
    methodology: str
    status: DataStatus
    retrieved_at: str  # ISO-8601 UTC timestamp — when *this build* actually fetched/computed it

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d
