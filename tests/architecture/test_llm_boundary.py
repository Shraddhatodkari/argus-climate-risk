"""Architecture-enforcement tests: 'deterministic engine computes, LLM explains'
(item #10 of the Enterprise spec).

    Hazard data -> DETERMINISTIC ENGINE -> RESULTS -> LLM -> Explanation only

Two things are enforced here, not just documented:

1. STATIC: no module inside the deterministic engine (geospatial/, financial/,
   analytics/, ingestion/, explainability/, common/ other than llm.py itself) is
   even allowed to import the LLM layer (common.llm) or an LLM-calling agent
   (agents.narrative_agent, agents.rag_agent). If a future change tried to route a
   risk number through an LLM call, this test fails at import-scan time, before any
   behavioural test would even run.

2. BEHAVIOURAL: narrative_agent.draft_for_district's numeric-consistency and
   citation-grounding checks actually reject output from a misbehaving LLM backend
   that invents a number or a citation the deterministic engine never produced --
   this is the runtime backstop for the same guarantee, using a deliberately
   'hallucinating' fake LLM client.

See docs/architecture.md's "Deterministic engine vs LLM" section for the full
design rationale this test is enforcing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from argus.agents.narrative_agent import GroundingError, draft_for_district
from argus.common.llm import LLMClient, LLMResponse

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "argus"

# Every module under these packages is part of the deterministic engine and must
# never import the LLM layer or an LLM-calling agent.
DETERMINISTIC_PACKAGES = ["geospatial", "financial", "analytics", "ingestion", "explainability"]

FORBIDDEN_IMPORT_PREFIXES = (
    "argus.common.llm",
    "argus.agents.narrative_agent",
    "argus.agents.rag_agent",
)


def _imported_module_names(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _all_python_files(package: str) -> list[Path]:
    return sorted((SRC_ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("package", DETERMINISTIC_PACKAGES)
def test_deterministic_package_never_imports_the_llm_layer(package):
    offenders = []
    for py_file in _all_python_files(package):
        imported = _imported_module_names(py_file)
        for name in imported:
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES):
                offenders.append((str(py_file.relative_to(SRC_ROOT.parents[1])), name))
    assert not offenders, (
        f"Deterministic-engine package {package!r} must never import the LLM layer, but found: {offenders}. "
        "The composite hazard score, exposure/impact figures, and stress-test/concentration numbers must be "
        "computable with zero LLM involvement — see docs/architecture.md."
    )


def test_common_config_and_provenance_never_import_the_llm_layer():
    for filename in ("config.py", "provenance.py", "schemas.py", "audit.py"):
        py_file = SRC_ROOT / "common" / filename
        imported = _imported_module_names(py_file)
        offenders = [n for n in imported if any(n == p or n.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES)]
        assert not offenders, f"{py_file} must never import the LLM layer, found: {offenders}"


class _HallucinatingLLM(LLMClient):
    """A deliberately misbehaving backend: invents a composite score the
    deterministic engine never computed, to prove the reviewer-facing guarantee
    ('every number is verified against the engine's actual output') is enforced at
    runtime, not just by convention."""

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="This district shows a composite climate physical-hazard score of 999.9/100, which is fabricated.",
            backend="hallucinating-test-double",
        )


class _CitationSpoofingLLM(LLMClient):
    """A deliberately misbehaving backend: cites a document ID that was never
    actually retrieved for this district."""

    def __init__(self, real_score: float) -> None:
        self.real_score = real_score

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=(
                f"This district shows a composite climate physical-hazard score of {self.real_score:.1f}/100. "
                "Per [FABRICATED-DOC-ID-NOT-RETRIEVED §Some Section], entities must comply."
            ),
            backend="citation-spoofing-test-double",
        )


def _demo_row() -> dict:
    return {
        "district": "Cuttack",
        "state": "Odisha",
        "composite_hazard_score": 46.09,
        "flood_component": 22.05,
        "drought_component": 14.53,
        "cyclone_component": 88.08,
        "heat_component": 67.22,
        "climate_exposed_exposure_inr_cr": 12345.6,
        "primary_hazard": "cyclone",
        "most_affected_sector": "Agriculture",
        "is_high_risk": True,
    }


def test_hallucinated_composite_score_is_rejected_before_reaching_review(monkeypatch):
    monkeypatch.setattr(
        "argus.agents.narrative_agent.get_llm_client",
        lambda context=None: _HallucinatingLLM(),
    )
    with pytest.raises(GroundingError):
        draft_for_district(_demo_row())


def test_spoofed_citation_is_rejected_before_reaching_review(monkeypatch):
    row = _demo_row()
    monkeypatch.setattr(
        "argus.agents.narrative_agent.get_llm_client",
        lambda context=None: _CitationSpoofingLLM(real_score=row["composite_hazard_score"]),
    )
    with pytest.raises(GroundingError):
        draft_for_district(row)


def test_the_default_template_backend_passes_both_checks():
    # Sanity check: the checks above aren't just rejecting everything -- the
    # deterministic TemplateLLM backend (used by default and in the rest of the
    # test suite) must still pass cleanly.
    draft = draft_for_district(_demo_row())
    assert draft.llm_backend == "template"
    assert "46.1" in draft.text
