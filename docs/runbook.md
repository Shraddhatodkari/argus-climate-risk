# Runbook

## Setup

```bash
pip install -e ".[dev]"
```

## Inspecting each stage of the pipeline directly

Every module is runnable on its own (`python -m argus.<module>`) so you can inspect
its output in isolation while developing:

```bash
python -m argus.agents.data_agent            # validate the district registry
python -m argus.geospatial.flood_index        # per-district satellite flood-extent score
python -m argus.ingestion.cyclone_imd         # real IMD cyclone best-track district exposure
python -m argus.ingestion.rainfall_imd        # real IMD rainfall-derived flood/drought proxies
python -m argus.ingestion.heat_imd            # real IMD station heat-exposure scores
python -m argus.ingestion.financial_exposure  # real NABARD x Census public exposure estimate
python -m argus.geospatial.hazard_scoring     # composite hazard scores (4 real components)
python -m argus.geospatial.exposure_engine    # full pipeline: hazard -> exposure -> sector -> impact
python -m argus.financial.impact_engine       # Hazard x Exposure x Vulnerability = Physical Impact
python -m argus.financial.stress_testing      # 4 climate stress scenarios
python -m argus.financial.translation         # financial-risk translation summary
python -m argus.analytics.concentration       # concentration insight sentences
python -m argus.geospatial.asset_exposure     # asset/location-level physical impact
python -m argus.rag.chunking                  # regulatory corpus chunks
python -m argus.rag.retriever                 # sample retrieval queries
python -m argus.rag.eval                      # RAG evaluation report (recall@3, abstention rate)
python -m argus.explainability.feature_contributions
python -m argus.augmentation.diffusion_augment  # classic tile augmentation demo
```

## Running the full pipeline end to end

```bash
python -m argus.agents.orchestrator
```

Registers one `run_id` in the SQLite audit log (`data/processed/audit_log.db`),
drafts one report item per district, writes each through the explicit lifecycle
(`generated → evidence_validated → submitted_for_review`), and prints which
districts are flagged high-risk. Nothing is auto-approved — items stay in
`audit.pending_review()` (defaults to the latest run only) until a human calls
`approve()`/`edit()`/`reject()`, and `publish()` is only meaningful after `approve()`
(see `agents/reviewer_gate.py`, or use the Review Queue / Decision Workflow tabs in
the Streamlit console below). Run it twice in a row and the pending queue still only
ever shows the latest run's items — `audit.history(run_id=...)` still sees every
past run in full.

## Running the analyst console

```bash
streamlit run app/streamlit_app.py
```

## Starting the MCP tool server

```bash
python -m argus.mcp_server.server
```

Point any MCP client (Claude Desktop's config, `claude mcp add`, or a custom client
using the `mcp` SDK) at this process over stdio.

## Switching the LLM backend

```bash
export ARGUS_LLM_MODE=template   # default — deterministic, zero network, zero cost
export ARGUS_LLM_MODE=ollama     # requires: ollama pull qwen2.5:3b-instruct && ollama serve
export ARGUS_LLM_MODE=claude     # requires: export ANTHROPIC_API_KEY=...
```

## Re-indexing the regulatory corpus

Drop a new `.txt` file into `data/regulatory_corpus/` following the existing
`[SOURCE: ...]` / `[DOC_ID: ...]` / `[SECTION: ...]` markup (see `rag/chunking.py`'s
docstring) — `TfidfRetriever()` re-indexes automatically on next construction, no
separate build step needed. After adding real source documents, re-run the eval
harness and update `docs/evaluation-report.md`:

```bash
python -m argus.rag.eval
```

## Quality gates (mirrors `.github/workflows/ci.yml`)

```bash
ruff check .
mypy src/argus
pip-audit
pytest -q --cov=argus --cov-report=term-missing
```

## Incident response

If the Narrative Agent's `_verify_numeric_consistency` or `_verify_citation` raises a
`GroundingError`, that item never reached the reviewer queue — check
`docs/evaluation-report.md`'s grounding-guarantee section, don't patch around the
rejection. If `agents/data_agent.py` raises `DataValidationError`, a raw source file
failed its Pandera schema — fix the source data, don't relax the schema. If
`tests/architecture/test_llm_boundary.py` fails, a change added an LLM import to a
deterministic-engine module (`geospatial/`, `financial/`, `analytics/`,
`ingestion/`, `explainability/`) — move the LLM call out of that module rather than
suppress the test; see `docs/architecture.md`'s "Deterministic engine vs LLM"
section for why that boundary is non-negotiable.
