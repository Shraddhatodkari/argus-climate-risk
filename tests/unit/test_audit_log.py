"""Unit tests: append-only, run-scoped audit log (Enterprise build).

The Enterprise build fixes a genuine usability bug found in the V1 demo: repeated
demo executions left the same district's narrative appearing multiple times in the
audit log with no way to tell which run a row belonged to (e.g. Kozhikode,
Kanchipuram, Cuttack reappearing as records 9-11 after a second run). The fix is
twofold -- every event now carries a ``run_id`` (registered once per run via
``start_run``), and ``pending_review()`` / ``latest_status()`` default to the LATEST
run only, while ``history()`` can still see every run's full record when asked.

The lifecycle vocabulary also changed from the old flat "drafted/approved/edited/
rejected" to the explicit decision-workflow chain:

    generated -> evidence_validated -> submitted_for_review -> approved/edited/rejected -> published
"""

from argus.common.audit import AuditLog


def _start_run(audit: AuditLog, run_id: str = "run-test000001") -> str:
    return audit.start_run(
        model_version="argus-narrative-agent-v1.1",
        data_version="test-data-v1",
        calculation_version="argus-hazard-exposure-v2.0-enterprise",
        run_id=run_id,
    )


def test_record_and_history_roundtrip(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = _start_run(audit)
    audit.record(
        run_id=run_id, item_id="X::1", district="X", event="generated", actor="narrative_agent",
        content="draft text", llm_backend="template",
    )
    history = audit.history("X::1")
    assert len(history) == 1
    assert history[0].event == "generated"
    assert history[0].run_id == run_id


def test_events_are_appended_never_overwritten(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = _start_run(audit)
    audit.record(run_id=run_id, item_id="X::1", district="X", event="generated", actor="narrative_agent",
                 content="draft", llm_backend="template")
    audit.record(run_id=run_id, item_id="X::1", district="X", event="submitted_for_review", actor="narrative_agent",
                 content="draft", llm_backend="template")
    audit.record(run_id=run_id, item_id="X::1", district="X", event="approved", actor="reviewer:alex",
                 content="draft", llm_backend="n/a", reviewer="alex", decision="approved")
    history = audit.history("X::1")
    assert [e.event for e in history] == ["generated", "submitted_for_review", "approved"]


def test_unknown_event_rejected(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = _start_run(audit)
    try:
        audit.record(run_id=run_id, item_id="A", district="A", event="drafted",
                     actor="x", content="x", llm_backend="x")
        assert False, "should have raised on an unrecognized audit event ('drafted' is no longer valid)"
    except ValueError:
        pass


def test_pending_review_lists_items_awaiting_human_decision(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = _start_run(audit)
    audit.record(run_id=run_id, item_id="A", district="A", event="generated", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_id, item_id="A", district="A", event="evidence_validated", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_id, item_id="A", district="A", event="submitted_for_review", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_id, item_id="B", district="B", event="generated", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_id, item_id="B", district="B", event="submitted_for_review", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_id, item_id="B", district="B", event="approved", actor="reviewer:alex",
                 content="d", llm_backend="n/a", reviewer="alex", decision="approved")
    assert audit.pending_review() == ["A"]


def test_pending_review_defaults_to_the_latest_run_only(tmp_path):
    """The direct fix for the user's item #12 complaint: repeated runs must not make
    earlier runs' districts reappear as if they were still pending in the current run."""
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_1 = _start_run(audit, "run-first000001")
    for district in ("Kozhikode", "Kanchipuram", "Cuttack"):
        audit.record(run_id=run_1, item_id=district, district=district, event="generated",
                     actor="narrative_agent", content="d", llm_backend="template")
        audit.record(run_id=run_1, item_id=district, district=district, event="submitted_for_review",
                     actor="narrative_agent", content="d", llm_backend="template")
    assert sorted(audit.pending_review(run_id=run_1)) == ["Cuttack", "Kanchipuram", "Kozhikode"]

    run_2 = _start_run(audit, "run-second000001")
    audit.record(run_id=run_2, item_id="Cuttack", district="Cuttack", event="generated",
                 actor="narrative_agent", content="d", llm_backend="template")
    audit.record(run_id=run_2, item_id="Cuttack", district="Cuttack", event="submitted_for_review",
                 actor="narrative_agent", content="d", llm_backend="template")

    # Only run_2's own district is pending under the (default) latest run --
    # run_1's Kozhikode/Kanchipuram/Cuttack rows do not leak through as duplicates.
    assert audit.pending_review() == ["Cuttack"]
    # But the full history across both runs is still queryable when explicitly asked.
    assert len(audit.history(run_id=run_1)) == 6
    assert len(audit.history(run_id=run_2)) == 2
    assert len(audit.history("Cuttack")) == 4  # 2 from run_1 + 2 from run_2


def test_latest_status_is_scoped_per_run(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_1 = _start_run(audit, "run-first000002")
    audit.record(run_id=run_1, item_id="A", district="A", event="generated", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_1, item_id="A", district="A", event="submitted_for_review", actor="narrative_agent",
                 content="d", llm_backend="template")
    audit.record(run_id=run_1, item_id="A", district="A", event="rejected", actor="reviewer:alex",
                 content="d", llm_backend="n/a", reviewer="alex", decision="rejected")

    run_2 = _start_run(audit, "run-second000002")
    audit.record(run_id=run_2, item_id="A", district="A", event="generated", actor="narrative_agent",
                 content="d", llm_backend="template")

    assert audit.latest_status(run_id=run_1) == {"A": "rejected"}
    assert audit.latest_status(run_id=run_2) == {"A": "generated"}
    assert audit.latest_status() == {"A": "generated"}  # defaults to latest run


def test_start_run_records_model_data_calculation_versions(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = audit.start_run(
        model_version="argus-narrative-agent-v1.1",
        data_version="argus-data-v2.0-enterprise",
        calculation_version="argus-hazard-exposure-v2.0-enterprise",
    )
    assert audit.latest_run_id() == run_id
    runs = audit.runs()
    assert len(runs) == 1
    assert runs[0].run_id == run_id
    assert runs[0].model_version == "argus-narrative-agent-v1.1"
    assert runs[0].data_version == "argus-data-v2.0-enterprise"
    assert runs[0].calculation_version == "argus-hazard-exposure-v2.0-enterprise"


def test_source_documents_round_trip_through_history(tmp_path):
    audit = AuditLog(db_path=tmp_path / "audit.db")
    run_id = _start_run(audit)
    audit.record(
        run_id=run_id, item_id="A", district="A", event="evidence_validated", actor="rag_agent",
        content="d", llm_backend="template", source_documents=["RBI-2024-CLIMATE-DISC#S4.2"],
    )
    history = audit.history("A")
    assert history[0].source_documents == '["RBI-2024-CLIMATE-DISC#S4.2"]'
