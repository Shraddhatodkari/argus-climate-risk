"""Central configuration.

Everything here has a working, zero-cost default: the pipeline runs fully offline
(ARGUS_LLM_MODE=template) with no API keys and no local model server required.
Set ARGUS_LLM_MODE=ollama or ARGUS_LLM_MODE=claude to opt into a real LLM backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CORPUS_DIR = DATA_DIR / "regulatory_corpus"
GOLD_DIR = DATA_DIR / "gold"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIT_DB_PATH = PROCESSED_DIR / "audit_log.db"

# District hazard threshold above which a district is flagged high-risk for review.
# Recalibrated against the real-data composite score distribution (Enterprise
# build, 8-district demo set): mean ~30.2, std ~9.4 -- 38.0 sits at ~mean + 0.8
# std, flagging the top tier (2 of 8 districts) rather than an arbitrary round
# number. Production deployments should still recalibrate this against real
# historical loss/claims data rather than a distributional heuristic — tracked as
# a backtesting task in docs/evaluation-report.md.
HIGH_RISK_THRESHOLD = 38.0

# A human-readable label for "which real data snapshots this run used" -- recorded
# on every audit-log run so a reviewer can tell, months later, exactly which
# dataset vintage a published figure was computed against. Bump this whenever any
# file under data/real/ is refreshed.
DATA_VERSION = (
    "data/real snapshot 2026-09-16: IMD RSMC New Delhi cyclone best-track (1982-2026); "
    "IMD sub-divisional rainfall (1901-2017); IMD station climatological normals; "
    "NABARD state credit potential (FY2025-26 to FY2026-27, per-state); Census 2011 population"
)

# Composite hazard score weights — must sum to 1.0. See docs/architecture.md for the
# methodology note and NGFS/BIS references this is modelled on. Recalibrated when
# extreme-heat was added as a fourth real hazard component (Enterprise build):
# flood keeps the largest weight (India's most damaging and most frequent physical
# hazard by historical loss count per NDMA/IMD summaries), heat is intentionally
# given real weight rather than tacked on at a token 5-10% just because it's the
# newest signal.
HAZARD_WEIGHTS = {
    "flood": 0.35,     # combines satellite NDWI water fraction + IMD rainfall-derived flood proxy
    "drought": 0.20,   # IMD rainfall-derived drought proxy
    "cyclone": 0.25,   # IMD best-track derived cyclone exposure
    "heat": 0.20,      # IMD station climatological extreme-heat normal
}

# Sector vulnerability weights: how much of a given hazard's score actually
# translates into financial impact for that sector. Expert-assigned, consistent
# with NGFS/TCFD physical-risk sector-sensitivity categorisation (agriculture is
# drought/flood sensitive; MSME/industry is flood + operational-disruption
# sensitive; other priority sectors, which in the NABARD split used here are
# mostly housing/real-estate-adjacent, are flood/heat sensitive) — not fitted to
# historical claims data in this build. See financial/vulnerability.py.
SECTOR_VULNERABILITY = {
    "Agriculture": {"flood": 0.90, "drought": 1.00, "cyclone": 0.70, "heat": 0.60},
    "MSME": {"flood": 0.80, "drought": 0.30, "cyclone": 0.60, "heat": 0.50},
    "Other": {"flood": 0.60, "drought": 0.20, "cyclone": 0.50, "heat": 0.70},
}

# Representative national priority-sector composition (NABARD, dated 2024-03-05,
# see data/real/nabard_sector_composition.csv) — used to split a district's public
# exposure estimate into sectors wherever a state-specific split isn't available.
DEFAULT_SECTOR_SPLIT = {"Agriculture": 0.52, "MSME": 0.39, "Other": 0.09}

# Climate stress-test scenarios: disclosed multiplicative shocks applied to each
# hazard component's score, and to the share of a sector's exposure treated as
# "affected" under that scenario. These are illustrative severity steps loosely
# aligned with the shape of NGFS physical-risk scenario framing (orderly /
# disorderly / hot-house-world), NOT NGFS's own calibrated output -- every
# dashboard view built on these says so and calls the results model estimates,
# never actual or projected bank losses. See financial/stress_testing.py.
STRESS_SCENARIOS = {
    "Baseline": {"hazard_multiplier": 1.00, "exposure_affected_share": 0.35},
    "Moderate Stress": {"hazard_multiplier": 1.25, "exposure_affected_share": 0.55},
    "Severe Stress": {"hazard_multiplier": 1.60, "exposure_affected_share": 0.75},
    "Long-Term Scenario": {"hazard_multiplier": 2.00, "exposure_affected_share": 0.90},
}
# Loss-given-hazard range applied on top of "affected" exposure to get an impact
# range rather than a false-precision point estimate -- again a disclosed
# assumption, not a fitted parameter.
IMPACT_RATE_RANGE = (0.08, 0.22)


@dataclass(frozen=True)
class Settings:
    llm_mode: str = os.environ.get("ARGUS_LLM_MODE", "template")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    retrieval_top_k: int = int(os.environ.get("ARGUS_RETRIEVAL_TOP_K", "3"))
    # Calibrated against data/gold/rag_qa_gold.jsonl + adversarial_qa.jsonl (see
    # tests/rag_eval/ and docs/evaluation-report.md): 0.15 gives 100% doc-level
    # recall@3 on the gold set and 80% correct abstention on the adversarial set with
    # the TF-IDF backend. The one known false-positive is documented in
    # docs/evaluation-report.md, with the semantic-rag extra as its fix.
    retrieval_min_score: float = float(os.environ.get("ARGUS_RETRIEVAL_MIN_SCORE", "0.15"))


SETTINGS = Settings()
