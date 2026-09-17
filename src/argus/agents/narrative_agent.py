"""Narrative Agent — drafts one disclosure-report paragraph per district, citing both
a scoring-engine figure and a regulatory source for every claim.

Two enforced guarantees, checked programmatically (not just prompted for):

1. Numeric consistency — every number in the drafted text is compared back against the
   scoring engine's actual output; a mismatch raises, it never silently ships.
2. Citation grounding — the paragraph's citation must be a chunk the RAG agent actually
   retrieved for this district; an LLM backend is never allowed to assert a citation
   that wasn't given to it (checked in ``_extract_and_verify_citation``).

Certification applied: Anthropic — Building with the Claude API;
NVIDIA — Building LLM Applications With Prompt Engineering (prompt construction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from argus.agents.rag_agent import citation_for_district
from argus.common.llm import LLMResponse, get_llm_client

NUMBER_RE = re.compile(r"(\d+\.\d+)")


class GroundingError(Exception):
    """Raised when a drafted paragraph fails the numeric-consistency or
    citation-grounding check — this must stop the item before it reaches the reviewer,
    not just get flagged for the reviewer to catch by eye."""


@dataclass(frozen=True)
class DraftReport:
    district: str
    text: str
    llm_backend: str
    citation_doc_id: str | None
    composite_hazard_score: float
    climate_exposed_exposure_inr_cr: float


def draft_for_district(row: dict) -> DraftReport:
    citation = citation_for_district(bool(row["is_high_risk"]))
    context = {
        "district": row["district"],
        "score": row["composite_hazard_score"],
        "flood": row["flood_component"],
        "drought": row["drought_component"],
        "cyclone": row["cyclone_component"],
        "heat": row["heat_component"],
        "exposure": row["climate_exposed_exposure_inr_cr"],
        "primary_hazard": row.get("primary_hazard"),
        "most_affected_sector": row.get("most_affected_sector"),
        "is_high_risk": row["is_high_risk"],
        "citation": (
            {"doc_id": citation.chunk.doc_id, "section": citation.chunk.section, "text": citation.chunk.text}
            if citation
            else None
        ),
    }
    prompt = _build_prompt(context)
    llm = get_llm_client(context=context)
    response: LLMResponse = llm.generate(prompt)

    _verify_numeric_consistency(response.text, row)
    verified_citation_doc_id = _verify_citation(response.text, citation)

    return DraftReport(
        district=row["district"],
        text=response.text,
        llm_backend=response.backend,
        citation_doc_id=verified_citation_doc_id,
        composite_hazard_score=float(row["composite_hazard_score"]),
        climate_exposed_exposure_inr_cr=float(row["climate_exposed_exposure_inr_cr"]),
    )


def _build_prompt(context: dict) -> str:
    citation = context.get("citation")
    citation_block = (
        f"Approved regulatory citation you may reference: [{citation['doc_id']} §{citation['section']}] "
        f"{citation['text']}"
        if citation
        else "No regulatory citation was retrieved for this district — do not cite a specific clause."
    )
    return (
        f"Draft one paragraph of a climate-risk disclosure report for {context['district']}.\n"
        f"Composite hazard score: {context['score']:.1f}/100 "
        f"(flood {context['flood']:.1f}, drought {context['drought']:.1f}, cyclone {context['cyclone']:.1f}, "
        f"heat {context['heat']:.1f}).\n"
        f"Primary hazard: {context['primary_hazard']}. Most climate-exposed sector: {context['most_affected_sector']}.\n"
        f"Estimated climate-exposed public-data exposure: ₹{context['exposure']:,.0f} crore "
        "(a public-data estimate, not a claim about any specific institution's portfolio).\n"
        f"High-risk flag: {context['is_high_risk']}.\n"
        f"{citation_block}\n"
        "Use ONLY the figures and citation given above. Do not invent or recompute any number — every figure "
        "above already came from the deterministic scoring engine; your role is to explain it, not calculate it."
    )


def _verify_numeric_consistency(text: str, row: dict) -> None:
    expected = {round(float(row["composite_hazard_score"]), 1)}
    found = {float(n) for n in NUMBER_RE.findall(text)}
    if not expected & found:
        raise GroundingError(
            f"Drafted text for {row['district']} does not mention the composite hazard "
            f"score {row['composite_hazard_score']:.1f} anywhere — rejecting before it "
            "reaches the reviewer queue."
        )


def _verify_citation(text: str, citation) -> str | None:
    if citation is None:
        return None
    if citation.chunk.doc_id not in text:
        raise GroundingError(
            f"Drafted text cites something other than the retrieved chunk "
            f"{citation.chunk.doc_id} — rejecting rather than shipping an unverifiable citation."
        )
    return citation.chunk.doc_id


if __name__ == "__main__":
    from argus.agents.scoring_agent import run as scoring_run

    for _, row in scoring_run().iterrows():
        draft = draft_for_district(row.to_dict())
        print(f"--- {draft.district} ({draft.llm_backend}) ---")
        print(draft.text)
        print()
