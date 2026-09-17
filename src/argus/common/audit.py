"""Append-only audit log (SQLite) for the human-in-the-loop review gate and the
explicit decision-workflow lifecycle -- Enterprise build.

Two things changed from the V1 demo build, both direct responses to a genuine
usability problem found by manually inspecting the review queue and audit-log tabs
of a repeatedly-re-run demo: the same district's drafted narrative appeared several
times over, once per demo run, with no way to tell which run a row belonged to.

1. Every row now carries a ``run_id`` (one per ``python -m argus.agents.orchestrator``
   invocation, recorded once in the new ``runs`` table with its model/data/
   calculation version and start time). Nothing is deleted — every past run's full
   history is still queryable via ``history(run_id=...)`` — but ``pending_review()``
   and the Streamlit Review Queue default to the LATEST run, so re-running the demo
   doesn't make the audit trail look like duplicate/broken data. One analysis run
   now means one run_id means one current result set, exactly as a production
   deployment needs.

2. The lifecycle is explicit and matches the workflow a real risk-review process
   follows, not just "drafted -> approved/edited/rejected":

       generated -> evidence_validated -> submitted_for_review -> approved/rejected -> published

   Every event records model_version, data_version, calculation_version,
   source_documents (the citations that grounded it), reviewer and decision where
   applicable -- everything the Data Lineage page and the "Decision Workflow" status
   view need to show a full, auditable chain from computation to publication.

Certification applied: Mastercard — Advisors & Consulting Services Job Simulation
(governance/audit design).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from argus.common.config import AUDIT_DB_PATH

LIFECYCLE_EVENTS = (
    "generated",
    "evidence_validated",
    "submitted_for_review",
    "approved",
    "edited",
    "rejected",
    "published",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    model_version TEXT NOT NULL,
    data_version TEXT NOT NULL,
    calculation_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    district TEXT NOT NULL,
    event TEXT NOT NULL,
    actor TEXT NOT NULL,
    content TEXT NOT NULL,
    llm_backend TEXT NOT NULL,
    citation_doc_id TEXT,
    source_documents TEXT,
    reviewer TEXT,
    decision TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""


@dataclass(frozen=True)
class AuditEvent:
    id: int
    run_id: str
    item_id: str
    district: str
    event: str
    actor: str
    content: str
    llm_backend: str
    citation_doc_id: str | None
    source_documents: str | None
    reviewer: str | None
    decision: str | None
    created_at: float


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    started_at: float
    model_version: str
    data_version: str
    calculation_version: str


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


class AuditLog:
    def __init__(self, db_path: Path | str = AUDIT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def start_run(
        self, *, model_version: str, data_version: str, calculation_version: str, run_id: str | None = None
    ) -> str:
        run_id = run_id or new_run_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, model_version, data_version, calculation_version) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, time.time(), model_version, data_version, calculation_version),
            )
        return run_id

    def latest_run_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def runs(self) -> list[RunInfo]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        return [RunInfo(**dict(r)) for r in rows]

    def record(
        self,
        *,
        run_id: str,
        item_id: str,
        district: str,
        event: str,
        actor: str,
        content: str,
        llm_backend: str,
        citation_doc_id: str | None = None,
        source_documents: list[str] | None = None,
        reviewer: str | None = None,
        decision: str | None = None,
    ) -> None:
        if event not in LIFECYCLE_EVENTS:
            raise ValueError(f"Unknown audit event: {event!r}; expected one of {LIFECYCLE_EVENTS}")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (run_id, item_id, district, event, actor, content, llm_backend, "
                "citation_doc_id, source_documents, reviewer, decision, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    item_id,
                    district,
                    event,
                    actor,
                    content,
                    llm_backend,
                    citation_doc_id,
                    json.dumps(source_documents) if source_documents is not None else None,
                    reviewer,
                    decision,
                    time.time(),
                ),
            )

    def history(self, item_id: str | None = None, run_id: str | None = None) -> list[AuditEvent]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM audit_log WHERE 1=1"
            params: list = []
            if item_id:
                query += " AND item_id = ?"
                params.append(item_id)
            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)
            query += " ORDER BY created_at ASC"
            rows = conn.execute(query, params).fetchall()
        return [AuditEvent(**dict(r)) for r in rows]

    def pending_review(self, run_id: str | None = None) -> list[str]:
        """item_ids whose latest event within the given run (default: the latest
        run) is still 'submitted_for_review' (awaiting a human decision)."""
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT item_id, event FROM audit_log a
                WHERE run_id = ?
                AND created_at = (
                    SELECT MAX(created_at) FROM audit_log b WHERE b.item_id = a.item_id AND b.run_id = a.run_id
                )
                """,
                (run_id,),
            ).fetchall()
        return [r["item_id"] for r in rows if r["event"] == "submitted_for_review"]

    def latest_status(self, run_id: str | None = None) -> dict[str, str]:
        """item_id -> latest lifecycle event, scoped to one run (default: latest) --
        what a Decision Workflow status view reads directly."""
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return {}
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT item_id, event FROM audit_log a
                WHERE run_id = ?
                AND created_at = (
                    SELECT MAX(created_at) FROM audit_log b WHERE b.item_id = a.item_id AND b.run_id = a.run_id
                )
                """,
                (run_id,),
            ).fetchall()
        return {r["item_id"]: r["event"] for r in rows}
