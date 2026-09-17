"""Regulatory RAG Agent — answers regulatory questions with a citation, or explicitly
abstains. This is the only agent allowed to assert what a regulation says; every other
agent that needs a regulatory citation calls through here rather than querying the
retriever directly, so the abstention behaviour is enforced in exactly one place.

Enterprise build: ``ask()`` returns a structured ``RegulatoryIntelligence`` record —
Question -> Applicable requirement -> Regulator -> Document -> Publication date ->
Section -> Evidence -> Applicability -> Human review flag — rather than a single
answer_text blob, so the Regulatory Intelligence tab can render each field
separately and a reviewer can see exactly which document/section/date backs a
claim without parsing prose. The abstain-if-no-evidence behaviour is unchanged and
still the load-bearing guarantee: ``grounded=False`` (or a low-confidence match)
always sets ``requires_human_review=True`` rather than presenting a guess as fact.

Certification applied: Anthropic — Building with the Claude API (this agent is where
the pluggable LLM client from common/llm.py is used for free-text regulatory Q&A, as
opposed to the template-only Narrative Agent).
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.rag.retriever import RetrievalResult, TfidfRetriever

# One retriever instance is reused across calls — re-fitting TF-IDF per query would be
# wasteful and the corpus doesn't change within a process lifetime.
_retriever: TfidfRetriever | None = None

# A confidence score below this, even when the retriever technically returns a
# match, is still routed for human review rather than presented as a confident
# answer — distinct from (and stricter than) SETTINGS.retrieval_min_score, which
# only gates whether a match is returned at all.
HUMAN_REVIEW_CONFIDENCE_FLOOR = 0.30

# Lightweight, per-document applicability notes -- who the document actually applies
# to, straight from each corpus file's own "Phased Applicability" / "Purpose"
# section where one exists. Not a substitute for reading the source document; a
# short, scannable field for the Regulatory Intelligence tab.
DOCUMENT_APPLICABILITY = {
    "RBI-2024-CLIMATE-DISC": "Scheduled Commercial Banks initially; phased extension to NBFCs and other regulated entities above defined thresholds (comply-or-explain).",
    "BIS-2024-CLIMATE-DISC": "Internationally active banks (voluntary; complements, does not replace, national supervisor requirements).",
    "NGFS-SCENARIO-GUIDANCE": "Central banks, supervisors, and financial institutions using NGFS scenarios for climate stress-testing and scenario analysis.",
}


def _get_retriever() -> TfidfRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TfidfRetriever()
    return _retriever


@dataclass(frozen=True)
class RegulatoryAnswer:
    """Retained for backward compatibility with existing tests/callers that only
    need the citation + a flat answer_text."""

    question: str
    grounded: bool
    citation: RetrievalResult | None
    answer_text: str


@dataclass(frozen=True)
class RegulatoryIntelligence:
    question: str
    grounded: bool
    applicable_requirement: str
    regulator: str
    document: str
    publication_date: str
    section: str
    evidence: str
    applicability: str
    confidence_score: float
    requires_human_review: bool


def ask(question: str) -> RegulatoryAnswer:
    retriever = _get_retriever()
    best = retriever.best(question)
    if best is None:
        return RegulatoryAnswer(
            question=question,
            grounded=False,
            citation=None,
            answer_text=(
                "The indexed RBI/BIS/NGFS corpus does not contain a confident answer to "
                "this question. Route to a human compliance reviewer rather than guessing."
            ),
        )
    return RegulatoryAnswer(
        question=question,
        grounded=True,
        citation=best,
        answer_text=f"[{best.chunk.doc_id} §{best.chunk.section}] {best.chunk.text}",
    )


def ask_structured(question: str) -> RegulatoryIntelligence:
    """The structured Regulatory Intelligence form of ``ask()`` -- same retrieval
    and the same abstain-if-no-evidence guarantee, decomposed into the fields a
    Regulatory Intelligence UI or a downstream report actually needs, each
    individually inspectable rather than baked into one prose string."""
    retriever = _get_retriever()
    best = retriever.best(question)

    if best is None:
        return RegulatoryIntelligence(
            question=question,
            grounded=False,
            applicable_requirement="No matching requirement found in the indexed corpus.",
            regulator="n/a",
            document="n/a",
            publication_date="n/a",
            section="n/a",
            evidence="",
            applicability="n/a",
            confidence_score=0.0,
            requires_human_review=True,
        )

    chunk = best.chunk
    return RegulatoryIntelligence(
        question=question,
        grounded=True,
        applicable_requirement=chunk.section.replace("_", " "),
        regulator=chunk.regulator,
        document=chunk.doc_id,
        publication_date=chunk.published,
        section=chunk.section,
        evidence=chunk.text,
        applicability=DOCUMENT_APPLICABILITY.get(chunk.doc_id, "Not specified in this corpus — verify against the source document."),
        confidence_score=round(best.score, 3),
        requires_human_review=best.score < HUMAN_REVIEW_CONFIDENCE_FLOOR,
    )


def citation_for_district(is_high_risk: bool) -> RetrievalResult | None:
    """Returns the regulatory citation for a district's risk status — used by the
    Narrative Agent so every drafted paragraph is grounded.

    This is a deterministic lookup, not a TF-IDF query, and that's deliberate: which
    clause applies to a high-risk vs. a within-tolerance district is a known, fixed
    business rule, not an open question. An earlier version of this function asked the
    retriever "find the passage about physical risk exposure by geography" and relied
    on it ranking BIS-2024-CLIMATE-DISC's Physical Risk Disclosure section top — but on
    this small corpus, RBI-2024-CLIMATE-DISC's Metrics and Targets section shares
    enough vocabulary ("physical risk", "geography", "disclose") to outrank it, so
    *every* district — high-risk or not — silently cited the same clause. Caught by
    inspecting the actual review-queue output, not by the retrieval eval (which never
    exercised this function). See docs/evaluation-report.md for the full note.
    """
    retriever = _get_retriever()
    if is_high_risk:
        citation = retriever.get("BIS-2024-CLIMATE-DISC", "Physical Risk Disclosure")
    else:
        citation = retriever.get("RBI-2024-CLIMATE-DISC", "Metrics and Targets")
    # Defensive fallback only — keeps the Narrative Agent from crashing if the corpus
    # is ever edited and a section is renamed/removed; should not trigger in normal use.
    return citation or retriever.best(
        "physical risk exposure disclosure" if is_high_risk else "climate risk metrics and targets"
    )


if __name__ == "__main__":
    for q in [
        "What should the Board disclose about climate risk oversight?",
        "What is the current repo rate?",
    ]:
        result = ask(q)
        print(f"Q: {q}\n  grounded={result.grounded}\n  {result.answer_text}\n")
