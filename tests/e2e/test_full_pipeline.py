"""End-to-end test: raw data -> validation -> hazard/exposure/vulnerability scoring
-> regulatory RAG -> narrative drafting -> human review queue -> decision workflow,
run against the frozen offline real-data snapshots (no live network required, no
live LLM required — ARGUS_LLM_MODE defaults to 'template').

Each call to ``run_pipeline`` is one analysis run with its own ``run_id`` (Enterprise
build) — that is what item #12 of the user's spec required: one analysis run -> one
run ID -> one scoped result set, so repeated executions never make the audit trail
look like duplicate/broken data."""

from argus.agents.orchestrator import run as run_pipeline
from argus.agents.reviewer_gate import approve
from argus.common.audit import AuditLog


def test_full_pipeline_runs_for_every_district_and_queues_for_review(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    orchestrator_run = run_pipeline(audit=audit)
    results = orchestrator_run.results

    assert len(results) == 8  # every district in the demo sample
    districts = {r.district for r in results}
    assert "Cuttack" in districts and "Latur" in districts  # a flood- and a drought-prone district both present

    # Every item must be sitting in the pending-review queue — nothing auto-publishes.
    pending = set(audit.pending_review(run_id=orchestrator_run.run_id))
    assert pending == {r.item_id for r in results}


def test_approving_a_pipeline_item_removes_it_from_the_queue(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    orchestrator_run = run_pipeline(audit=audit)
    results = orchestrator_run.results
    first = results[0]

    approve(first.item_id, first.district, first.draft.text, reviewer="test-cro",
            audit=audit, run_id=orchestrator_run.run_id)

    pending = audit.pending_review(run_id=orchestrator_run.run_id)
    assert first.item_id not in pending
    assert len(pending) == len(results) - 1


def test_high_risk_districts_are_a_strict_subset_flagged_correctly(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    orchestrator_run = run_pipeline(audit=audit)
    results = orchestrator_run.results
    high_risk = [r for r in results if r.is_high_risk]
    assert 0 < len(high_risk) < len(results)  # some, not all, not none — a real signal
    for r in high_risk:
        assert "HIGH RISK" in r.draft.text or str(round(r.draft.composite_hazard_score, 1)) in r.draft.text


def test_two_consecutive_runs_stay_cleanly_scoped(tmp_path):
    """Direct regression test for item #12: repeated runs must not leave stale
    districts from an earlier run looking pending in the latest run's queue."""
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_1 = run_pipeline(audit=audit)
    run_2 = run_pipeline(audit=audit)

    assert run_1.run_id != run_2.run_id
    # The default (latest-run) pending queue only ever shows this run's 8 items.
    assert len(audit.pending_review()) == 8
    assert set(audit.pending_review()) == {r.item_id for r in run_2.results}
    # But nothing was lost — both runs' full histories are still there.
    assert len(audit.runs()) == 2
