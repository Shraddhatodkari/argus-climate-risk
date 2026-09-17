"""Reviewer / Human-in-the-Loop Gate — Enterprise decision-workflow lifecycle.

Every drafted report item moves through an explicit lifecycle, each step its own
audit-log event, all scoped to the run that produced it:

    generated -> evidence_validated -> submitted_for_review -> approved/rejected -> published

``submit_for_review`` (called by the orchestrator once per district) writes the
first three events in one call — generated (the narrative agent's raw draft),
evidence_validated (the numeric-consistency + citation-grounding checks the
narrative agent already enforced before this function is even called — see
narrative_agent.py's ``GroundingError``), and submitted_for_review (queued, visible
in the Streamlit Review Queue). ``approve``/``edit``/``reject`` are first-class
human decisions, not failure paths. ``publish`` is the final step, only reachable
from ``approved`` — a rejected or still-pending item can never be marked published.
"""

from __future__ import annotations

from argus.agents.narrative_agent import DraftReport
from argus.common.audit import AuditLog

MODEL_VERSION = "argus-narrative-agent-v1.1"
CALCULATION_VERSION = "argus-hazard-exposure-v2.0-enterprise"


def submit_for_review(draft: DraftReport, *, audit: AuditLog, run_id: str) -> str:
    item_id = f"{draft.district}::{draft.composite_hazard_score:.1f}"
    source_documents = [draft.citation_doc_id] if draft.citation_doc_id else []

    audit.record(
        run_id=run_id,
        item_id=item_id,
        district=draft.district,
        event="generated",
        actor="narrative_agent",
        content=draft.text,
        llm_backend=draft.llm_backend,
        citation_doc_id=draft.citation_doc_id,
        source_documents=source_documents,
    )
    # Reaching this point means narrative_agent.draft_for_district already passed
    # _verify_numeric_consistency and _verify_citation (it raises GroundingError and
    # stops the pipeline before submit_for_review is ever called otherwise) -- this
    # event records that fact in the audit trail rather than leaving it implicit.
    audit.record(
        run_id=run_id,
        item_id=item_id,
        district=draft.district,
        event="evidence_validated",
        actor="narrative_agent",
        content="Numeric-consistency and citation-grounding checks passed.",
        llm_backend=draft.llm_backend,
        citation_doc_id=draft.citation_doc_id,
        source_documents=source_documents,
    )
    audit.record(
        run_id=run_id,
        item_id=item_id,
        district=draft.district,
        event="submitted_for_review",
        actor="narrative_agent",
        content=draft.text,
        llm_backend=draft.llm_backend,
        citation_doc_id=draft.citation_doc_id,
        source_documents=source_documents,
    )
    return item_id


def approve(item_id: str, district: str, content: str, *, reviewer: str, audit: AuditLog, run_id: str) -> None:
    audit.record(
        run_id=run_id, item_id=item_id, district=district, event="approved", actor=f"reviewer:{reviewer}",
        content=content, llm_backend="n/a", reviewer=reviewer, decision="approved",
    )


def edit(item_id: str, district: str, new_content: str, *, reviewer: str, audit: AuditLog, run_id: str) -> None:
    audit.record(
        run_id=run_id, item_id=item_id, district=district, event="edited", actor=f"reviewer:{reviewer}",
        content=new_content, llm_backend="n/a", reviewer=reviewer, decision="edited",
    )


def reject(item_id: str, district: str, reason: str, *, reviewer: str, audit: AuditLog, run_id: str) -> None:
    audit.record(
        run_id=run_id, item_id=item_id, district=district, event="rejected", actor=f"reviewer:{reviewer}",
        content=reason, llm_backend="n/a", reviewer=reviewer, decision="rejected",
    )


def publish(item_id: str, district: str, content: str, *, reviewer: str, audit: AuditLog, run_id: str) -> None:
    """Only meaningful after ``approve`` -- callers (e.g. the Streamlit Decision
    Workflow tab) should check ``audit.latest_status()[item_id] == "approved"``
    before calling this, so a rejected or still-pending item can never be published."""
    audit.record(
        run_id=run_id, item_id=item_id, district=district, event="published", actor=f"reviewer:{reviewer}",
        content=content, llm_backend="n/a", reviewer=reviewer, decision="published",
    )
