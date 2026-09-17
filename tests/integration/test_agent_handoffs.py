"""Integration tests: agent-to-agent handoffs (not full end-to-end — see tests/e2e/)."""

from argus.agents import scoring_agent
from argus.agents.narrative_agent import draft_for_district
from argus.agents.reviewer_gate import approve, reject, submit_for_review
from argus.common.audit import AuditLog


def test_scoring_agent_output_feeds_narrative_agent_without_kwargs_mismatch():
    row = scoring_agent.run().iloc[0].to_dict()
    draft = draft_for_district(row)
    assert draft.district == row["district"]
    assert draft.text  # non-empty


def test_rag_agent_citation_flows_into_narrative_agent():
    row = scoring_agent.run_for_district("Cuttack")
    draft = draft_for_district(row)
    # Cuttack is high-risk in the demo sample, so a citation should have been retrieved.
    assert draft.citation_doc_id is not None


def test_reviewer_gate_records_full_lifecycle(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = audit.start_run(model_version="test", data_version="test", calculation_version="test")
    row = scoring_agent.run().iloc[0].to_dict()
    draft = draft_for_district(row)

    item_id = submit_for_review(draft, audit=audit, run_id=run_id)
    assert item_id in audit.pending_review(run_id=run_id)

    approve(item_id, draft.district, draft.text, reviewer="test-reviewer", audit=audit, run_id=run_id)
    assert item_id not in audit.pending_review(run_id=run_id)

    history = audit.history(item_id)
    assert [e.event for e in history] == ["generated", "evidence_validated", "submitted_for_review", "approved"]


def test_reject_removes_item_from_pending_queue(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = audit.start_run(model_version="test", data_version="test", calculation_version="test")
    row = scoring_agent.run().iloc[1].to_dict()
    draft = draft_for_district(row)
    item_id = submit_for_review(draft, audit=audit, run_id=run_id)

    reject(item_id, draft.district, "needs a second source", reviewer="test-reviewer", audit=audit, run_id=run_id)
    assert item_id not in audit.pending_review(run_id=run_id)
    assert audit.history(item_id)[-1].event == "rejected"


def test_mcp_tools_match_underlying_agent_output():
    from argus.mcp_server.tools import get_hazard_exposure

    mcp_result = get_hazard_exposure("Cuttack")
    agent_result = scoring_agent.run_for_district("Cuttack")
    assert mcp_result["composite_hazard_score"] == agent_result["composite_hazard_score"]
    assert mcp_result["climate_exposed_exposure_inr_cr"] == agent_result["climate_exposed_exposure_inr_cr"]
    assert mcp_result["exposure_basis"] == "Public-data exposure estimate (not a specific institution's portfolio)"
