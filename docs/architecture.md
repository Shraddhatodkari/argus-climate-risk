# Architecture — Enterprise build

This describes what is **actually implemented and tested** in `src/argus/`, not just
planned. Every module named below exists and is exercised by the test suite (see
`docs/evaluation-report.md` for current numbers). Where a choice trades off against a
heavier alternative, that's called out explicitly with the reasoning and the
documented upgrade path.

This build answers the reframed business question the spec asks for: *where is a
portfolio financially vulnerable to physical climate risk, how large could the impact
be under different scenarios, why is it happening, and what should risk teams do?* —
not just "what's this district's hazard score."

## Pipeline

```
Real public data                Data Agent           Hazard & Exposure & Vulnerability
(IMD cyclone best-track,   →   (schema-validate  →   Agent (scoring_agent.py, wrapping
 IMD rainfall, IMD station      the district           geospatial/ + financial/):
 climatology, NABARD state      registry via              hazard_scoring  → composite score
 credit potential, Census       Pandera)                  financial_exposure → public exposure
 2011 population)                                         sector_allocation → sector split      ─┐
                                                            impact_engine → physical impact         │
                                                            stress_testing → 4 scenarios              │
Regulatory corpus  →  Regulatory RAG Agent ───────────────  concentration → rollups + insights       ├→ Narrative Agent → Reviewer/HITL gate
(RBI/BIS/NGFS,          (TF-IDF retrieval,                                                             │   (citation +      (SQLite audit log,
 paraphrased,            abstains below                                                                │    numeric checks)  run-scoped, explicit
 labeled)                confidence threshold,                                                         │                     lifecycle, never
                          structured output)                                                            ┘                    auto-publishes)

All of the above (hazard lookup, exposure lookup, regulatory search) is also exposed
as MCP tools (mcp_server/server.py) — the exact same functions the orchestrator and
the Streamlit console call, not a separate implementation.
```

## Agents (`src/argus/agents/`)

| Agent | Module | What it actually does |
|---|---|---|
| Data Agent | `data_agent.py` | Loads the district registry (identity + centroid only) and validates it against a Pandera schema (`common/schemas.py`); raises `DataValidationError` on any violation before scoring runs |
| Hazard & Exposure Agent | `scoring_agent.py` (wraps `geospatial/exposure_engine.py`) | `run()` returns the flat exposure table; `run_full_pipeline()` returns every intermediate table (hazard scores, sector exposure, impact detail, provenance) the deeper Streamlit tabs and the Data Lineage page need |
| Regulatory RAG Agent | `rag_agent.py` | `ask()` (flat) and `ask_structured()` (Question → Applicable requirement → Regulator → Document → Publication date → Section → Evidence → Applicability → Human review) — the only agent allowed to assert what a regulation says |
| Narrative Agent | `narrative_agent.py` | Drafts one paragraph per district via the pluggable LLM client; programmatically verifies the number and citation it used before returning — see "Deterministic engine vs LLM" below |
| Reviewer / HITL gate | `reviewer_gate.py` | Writes the explicit lifecycle (`generated → evidence_validated → submitted_for_review → approved/edited/rejected → published`) to the run-scoped audit log |

`orchestrator.py` wires these into one explicit pipeline function, `run()`, returning
an `OrchestratorRun(run_id, results, pipeline_output)`. It is a plain Python state
machine (a sequence of named steps sharing a context), not LangGraph — see
"Deliberate simplifications" below for why, and `pyproject.toml`'s `orchestration`
extra for the upgrade path.

## Deterministic engine vs LLM (item #10 — the most important boundary in this build)

```
Hazard data (IMD)  →  DETERMINISTIC ENGINE  →  RESULTS  →  LLM  →  Explanation only
Exposure data          (hazard_scoring,           (scores,
Vulnerability model      impact_engine,             exposure,
                          stress_testing,            impact,
                          concentration)             stress)
```

Every score, exposure figure, physical-impact number, stress-test result, and
concentration statistic is produced by a plain, auditable Python function — no LLM
call is ever in that path. This is enforced two ways, not just documented:

1. **Statically** — `tests/architecture/test_llm_boundary.py` AST-scans every module
   under `geospatial/`, `financial/`, `analytics/`, `ingestion/`, `explainability/`
   and asserts none of them imports `common.llm`, `agents.narrative_agent`, or
   `agents.rag_agent`. A future change that tried to route a risk number through an
   LLM call would fail this test at import-scan time.
2. **At runtime** — `narrative_agent.py`'s `_verify_numeric_consistency` and
   `_verify_citation` reject any drafted text whose score doesn't match the engine's
   actual output, or whose citation wasn't actually retrieved, and raise
   `GroundingError` before the item ever reaches the review queue. The same test file
   exercises this with deliberately misbehaving ("hallucinating"/"citation-spoofing")
   fake LLM backends to prove the rejection is real, not just prompted for.

The LLM layer is allowed to: explain a score in prose, summarize, answer regulatory
questions (retrieval-grounded, `rag_agent.ask_structured`), draft management
commentary, and identify evidence. It is never allowed to invent or calculate a core
risk number.

## Hazard scoring methodology (`geospatial/hazard_scoring.py`)

Four independently-sourced, real hazard signals combine into one 0–100 composite
score per district:

```
flood_component    = 0.6 * satellite NDWI flood-extent score (Proxy — synthetic tile,
                      see ingestion/copernicus.py) + 0.4 * IMD-rainfall-derived flood
                      proxy (Derived from Observed IMD sub-divisional rainfall)
drought_component  = IMD-rainfall-derived drought proxy (Derived, same source)
cyclone_component  = IMD best-track-derived cyclone exposure (Derived from Observed
                      IMD RSMC New Delhi records)
heat_component     = IMD station climatological extreme-heat exposure (Derived from
                      Observed IMD station normals)

composite_hazard_score = 0.35*flood + 0.20*drought + 0.25*cyclone + 0.20*heat
```

Three of the four components are Derived from real, cited, government-published
historical records (see `docs/data-sources.md`); the satellite signal remains a
disclosed Proxy (synthetic tile — this sandbox cannot reach a live Copernicus/
Sentinel-2 feed) and carries its own `DataProvenance` record
(`hazard_scoring.SATELLITE_FLOOD_PROVENANCE`) so it is never mistaken for an
observation.

This is still a transparent, linear, weighted-index methodology — deliberately, so
`explainability/feature_contributions.py` can decompose a score *exactly* rather than
approximate it with SHAP.

## Financial exposure, vulnerability, and physical impact (`financial/`, `ingestion/financial_exposure.py`)

```
Public financial exposure estimate            Institution portfolio exposure
(NABARD state credit potential apportioned      (a real bank/NBFC's actual book —
 by Census 2011 district population share)       NOT present in this build)
        ↓ financial/sector_allocation.py                    ↓ same swap-in interface
Sector exposure (Agriculture/MSME/Other),      ingestion.financial_exposure.
state-specific NABARD split where on file,     load_institution_portfolio(path=...)
else the representative national split                      ↓
        ↓ financial/vulnerability.py + financial/impact_engine.py
Physical Climate Impact = Hazard × Exposure × Vulnerability
```

`impact_engine.py`'s `compute_physical_impact` computes, per (district, sector,
hazard): `sensitivity_share = HAZARD_WEIGHTS[hazard] × (hazard_score/100) ×
vulnerability_weight(sector, hazard)`, then `physical_impact = sensitivity_share ×
sector_exposure`. Folding `HAZARD_WEIGHTS` (which sum to 1.0) into the per-hazard
term — rather than summing four independent, uncapped `(score × weight)` terms — is a
deliberate correctness choice: it guarantees `combined_sensitivity ∈ [0, 1]`, so a
district's climate-exposed exposure can never exceed its actual exposure. An earlier
draft of this module didn't do this and could show a sector's combined impact
exceeding 100% of its exposure; the fix (and why) is documented directly in that
module's own docstring, and `tests/unit/test_exposure_engine.py::
test_climate_exposed_exposure_never_exceeds_public_exposure` is the regression test.

`financial/stress_testing.py` applies the same bounded formula under four disclosed
scenarios (`common/config.py`'s `STRESS_SCENARIOS`: Baseline, Moderate Stress, Severe
Stress, Long-Term Scenario), each with a hazard multiplier and an exposure-affected
share, then an `IMPACT_RATE_RANGE` to produce a low–high estimated impact range
rather than a false-precision point figure. Every caller of this module must keep the
"model estimate, not an actual or projected loss" framing next to the numbers.

`financial/translation.py` composes the baseline impact and the Severe Stress
scenario into the one-row-per-district summary a risk committee actually reads
(Portfolio exposure, Climate-exposed exposure, Severe-stress exposure, Estimated
impact range, Primary hazard, Most affected sector).

`analytics/concentration.py` rolls the same impact detail up along State → District →
Sector → Hazard → Exposure and turns the largest concentration into a plain-English,
management-readable sentence (`top_concentration_insight`,
`sector_geography_insight`) — deterministic groupby + sort, no LLM involved in
deciding which concentration is largest.

`geospatial/asset_exposure.py` extends the same chain one level finer: Borrower/asset
→ lat/lon → district → the same four hazard components → the same sector
vulnerability weights → asset-level physical impact. The bundled
`data/samples/asset_locations.csv` is explicitly illustrative demo data (real,
individually-geolocated borrower data is confidential and cannot ship in a public
project) — a deploying institution swaps in its own file via
`load_asset_locations(path=...)`, and every downstream calculation runs unchanged.

## Regulatory RAG (`rag/`, `agents/rag_agent.py`)

`chunking.py` parses the bundled corpus (`data/regulatory_corpus/*.txt` —
paraphrased, clearly labeled summaries of the real RBI/BIS/NGFS publications, each
tagged with `[REGULATOR: ...]`, `[PUBLISHED: ...]`, `[DOC_URL: ...]`; see each file's
header and `docs/data-sources.md`) into doc/section-tagged chunks.

`retriever.py`'s `TfidfRetriever` is the default: TF-IDF + cosine similarity via
scikit-learn — no multi-GB download, millisecond queries on a CPU-only laptop, and
for a domain-specific corpus this size, retrieval quality is genuinely competitive
(see `docs/evaluation-report.md`: 100% gold recall@3). `embeddings.py`'s
`SemanticRetriever` implements the identical `retrieve()`/`best()` interface backed
by `sentence-transformers` + `ChromaDB`, gated behind the `semantic-rag` extra — a
drop-in swap, not a rewrite.

`rag_agent.ask_structured()` decomposes a query into the Regulatory Intelligence form
the spec asks for (Question → Applicable requirement → Regulator → Document →
Publication date → Section → Evidence → Applicability → Human review), with a
`HUMAN_REVIEW_CONFIDENCE_FLOOR` stricter than the retrieval minimum score — even a
technically-grounded match below that floor is flagged for human review rather than
presented as confident. If no evidence is found, the agent abstains rather than
guessing; this is the load-bearing guarantee and is unchanged from V1.

## LLM backend (`common/llm.py`)

Three real implementations behind one `LLMClient` interface — `TemplateLLM`
(deterministic default, what CI runs), `OllamaLLM` (local, zero-cost), `ClaudeLLM`
(Anthropic SDK, gated on `ANTHROPIC_API_KEY`). See "Deterministic engine vs LLM"
above for the enforcement of what this layer is and isn't allowed to do.

## Audit log & decision workflow (`common/audit.py`, `agents/reviewer_gate.py`)

SQLite, append-only (`INSERT`, never `UPDATE`). Enterprise build adds two things,
both direct fixes for problems found by inspecting a repeatedly-re-run demo:

1. **Run scoping.** Every pipeline invocation registers one `run_id` in a `runs`
   table (with `model_version`, `data_version`, `calculation_version`, start time).
   Every audit event is tagged with it. `pending_review()` and `latest_status()`
   default to the LATEST run only; nothing is deleted, and `history(run_id=...)`
   still sees every past run in full. This is the fix for "repeated demo runs made
   the same districts reappear as duplicate-looking records" (previously: Kozhikode,
   Kanchipuram, Cuttack reappearing as records 9–11 after a second run).
2. **Explicit lifecycle.** `generated → evidence_validated → submitted_for_review →
   approved/edited/rejected → published`, not a flat "drafted/approved". Every event
   records model/data/calculation version, the source documents that grounded it,
   the reviewer, and the decision — a full, auditable chain from computation to
   publication, which is what the Decision Workflow view in the Streamlit console
   and the Data Lineage page's STATUS field both read directly.

## Data Lineage (`app/streamlit_app.py`'s Data Lineage tab)

For any result, shows RESULT / INPUTS / DATA SOURCES (every dataset's
`DataProvenance` — dataset, organization, status, publication date, geographic
resolution, retrieved-at, source URL) / MODEL (calculation + narrative version
strings) / CALCULATION (the exact formula) / GENERATED (render timestamp) / STATUS
(the item's current decision-workflow state, if drafted in the current run). Every
field is read directly from the same `PipelineOutput.provenance` dict and
`AuditLog` the rest of the console uses — nothing is re-derived just for this page.

## MCP tool server (`mcp_server/`)

Built on the official `mcp` Python SDK's `FastMCP` high-level API. Three tools —
`get_hazard_exposure`, `list_high_risk_districts`, `search_regulatory_corpus` — call
directly into the same agent/geospatial/rag modules the orchestrator and the
Streamlit console use. `get_hazard_exposure`'s response explicitly labels its
exposure figures as a public-data estimate, not a specific institution's portfolio.
Run with `python -m argus.mcp_server.server` and point any MCP client at it over
stdio.

## Deliberate simplifications in this build (and the documented upgrade path)

| Choice | Why | Upgrade path |
|---|---|---|
| TF-IDF retrieval, not sentence-transformers | Zero multi-GB download, CPU-millisecond queries | `pyproject.toml`'s `semantic-rag` extra → `rag/embeddings.py` |
| Hand-rolled orchestrator, not LangGraph | Five sequential steps + one branch doesn't need a graph library yet | `orchestration` extra; step functions are already `dict → dict`, LangGraph-node-compatible |
| Plotly `scatter_map` + lat/lon centroids/points, not GeoPandas/Rasterio polygons | No GDAL toolchain required to install | `gis` extra; swap district centroids / asset points for real polygon joins |
| Synthetic satellite tile (Proxy, disclosed) for the flood signal's NDWI term | This sandbox can't reach a live Copernicus/Sentinel-2 feed | `ingestion/copernicus.py::fetch_live_tile` is the documented swap-in point |
| Illustrative demo asset/borrower locations (Proxy, disclosed) | Real individually-geolocated borrower data is confidential and cannot ship in a public project | `geospatial/asset_exposure.py::load_asset_locations(path=...)` |
| Public-data exposure estimate, not a real bank's portfolio | This is a public, non-proprietary project — no confidential loan book is available or appropriate to fabricate | `ingestion/financial_exposure.py::load_institution_portfolio(path=...)` |
| SQLite audit log, not a managed database | Zero infrastructure, real ACID guarantees, real SQL | Swap the connection string in `common/audit.py` for Postgres when this leaves a single analyst's laptop |
| Illustrative stress-scenario severity steps, not NGFS's own calibrated scenario output | No claims/loss data in this build to calibrate against | `common/config.py`'s `STRESS_SCENARIOS`/`IMPACT_RATE_RANGE`; replace with NGFS-calibrated parameters when available |

## Non-negotiables

- No AI-drafted report content reaches "published" status without a logged human
  approval (`reviewer_gate.py`; every item starts in `audit.pending_review()`, and
  `publish()` is only meaningful after `approve()`).
- Every regulatory claim in a drafted paragraph carries a citation that was actually
  retrieved, checked programmatically (`narrative_agent._verify_citation`).
- No hazard, exposure, or financial figure is computed by an LLM — see "Deterministic
  engine vs LLM" above, enforced by `tests/architecture/test_llm_boundary.py`.
- Every dataset this pipeline uses carries an explicit Observed/Derived/Modelled/
  Proxy status (`common/provenance.py`) — never presented without it.
- A district's climate-exposed exposure, severe-stress exposure, and asset-level
  physical impact can never exceed the exposure they're computed against — enforced
  by the `HAZARD_WEIGHTS`-bounded formula (see above) and its regression tests.
