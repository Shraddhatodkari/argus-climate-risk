"""Agent orchestrator — wires Data -> Hazard & Exposure -> Regulatory RAG ->
Narrative -> Reviewer gate into one explicit, testable pipeline, with an explicit
per-run identity (Enterprise build): every invocation starts one ``run_id`` in the
audit log's ``runs`` table (recording model/data/calculation version and start
time), and every audit event this run produces is tagged with it. That is what
fixes the "repeated demo executions make the audit trail look messy" problem --
each run is its own clearly-scoped set of results, and ``AuditLog.pending_review()``
/ ``latest_status()`` default to the most recent run rather than mixing every past
run's rows together.

Implemented as a plain, dependency-free state machine (a list of named steps over a
shared context dict) rather than pulling in LangGraph: for five sequential agents with
one conditional branch (the review gate), a graph library adds a dependency without
adding testability. ``orchestration`` in pyproject.toml documents the LangGraph
upgrade path for once the agent graph grows branches/loops that actually need it —
the step interface below (``dict -> dict``) is intentionally compatible with becoming
LangGraph nodes later without a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.agents import data_agent, scoring_agent
from argus.agents.narrative_agent import DraftReport, draft_for_district
from argus.agents.reviewer_gate import CALCULATION_VERSION, MODEL_VERSION, submit_for_review
from argus.common.audit import AuditLog
from argus.common.config import DATA_VERSION
from argus.geospatial.exposure_engine import PipelineOutput


@dataclass(frozen=True)
class PipelineResult:
    item_id: str
    district: str
    draft: DraftReport
    is_high_risk: bool


@dataclass(frozen=True)
class OrchestratorRun:
    run_id: str
    results: list[PipelineResult]
    pipeline_output: PipelineOutput


def run(*, audit: AuditLog | None = None) -> OrchestratorRun:
    audit = audit or AuditLog()
    run_id = audit.start_run(
        model_version=MODEL_VERSION, data_version=DATA_VERSION, calculation_version=CALCULATION_VERSION
    )

    # Step 1 — Data Agent: validate every raw source before scoring touches it.
    data_agent.run()

    # Step 2 — Hazard & Exposure Agent (full pipeline: hazard scores, sector
    # exposure, physical impact -- everything the Streamlit app's deeper tabs need).
    pipeline_output = scoring_agent.run_full_pipeline()
    exposure_df = pipeline_output.exposure_at_risk

    results: list[PipelineResult] = []
    for _, row in exposure_df.iterrows():
        row_dict = row.to_dict()

        # Step 3 (inside narrative_agent) + Step 4 — Regulatory RAG Agent + Narrative
        # Agent: draft a citation-grounded, numerically-verified paragraph.
        draft = draft_for_district(row_dict)

        # Step 5 — Reviewer / HITL gate: every draft is queued, never auto-published.
        item_id = submit_for_review(draft, audit=audit, run_id=run_id)

        results.append(
            PipelineResult(item_id=item_id, district=row_dict["district"], draft=draft,
                            is_high_risk=bool(row_dict["is_high_risk"]))
        )
    return OrchestratorRun(run_id=run_id, results=results, pipeline_output=pipeline_output)


if __name__ == "__main__":
    audit = AuditLog()
    orchestrator_run = run(audit=audit)
    for r in orchestrator_run.results:
        flag = "HIGH RISK" if r.is_high_risk else "ok"
        print(f"[{flag:9s}] {r.district:15s} item_id={r.item_id}  backend={r.draft.llm_backend}")
    print(f"\nRun ID: {orchestrator_run.run_id}")
    print(f"Pending human review (this run): {audit.pending_review(run_id=orchestrator_run.run_id)}")
