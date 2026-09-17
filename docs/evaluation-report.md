# Evaluation report

**Last measured:** this build (Enterprise), against the real data snapshots
(`data/real/`), the bundled demo fixtures (`data/samples/`), the regulatory corpus
(`data/regulatory_corpus/`), and the RAG gold set (`data/gold/`). Re-run the commands
below and update this file whenever the corpus, retriever, or scoring formula
changes — treat it as a running log, not a one-time report.

## Test suite

```
$ pytest -q --cov=argus --cov-report=term-missing
111 passed, 1 skipped in ~13s
TOTAL coverage: 83%
```

The one skip is `test_claude_backend_when_api_key_present`, which only runs when
`ANTHROPIC_API_KEY` is set — by design, so CI never depends on a paid key.

The jump from V1's 64 tests to 111 reflects the Enterprise rewrite: new real-data
ingestion modules (cyclone/rainfall/heat), the sector vulnerability/impact/stress-
testing/concentration/translation chain, asset-level exposure, the run-scoped audit
log's new lifecycle, and `tests/architecture/test_llm_boundary.py`'s static +
behavioural enforcement of the deterministic-engine/LLM boundary (item #10).

Reproduce: `pytest -q --cov=argus --cov-report=term-missing`

## Streamlit console smoke test

Not part of the `pytest` run above (Streamlit apps need a script-runner harness, not
plain pytest), but exercised the same way in CI-equivalent local verification via
`streamlit.testing.v1.AppTest`: loads `app/streamlit_app.py`, runs every one of its
11 tabs' code paths (tabs render unconditionally regardless of which is visually
selected), then drives the actual interactive flow — click "Run pipeline", approve an
item, ask a Regulatory Intelligence question, publish an approved item, move a
What-If severity slider, reject an item — asserting no exception at each step. All
verified clean on this build.

## RAG retrieval evaluation

Backend: `TfidfRetriever` (scikit-learn TF-IDF + cosine similarity), `min_score=0.15`,
`top_k=3`, against the 12-chunk bundled corpus (`data/regulatory_corpus/`).

| Metric | Result |
|---|---|
| Gold-set doc-level recall@3 (`data/gold/rag_qa_gold.jsonl`, 10 questions) | **10/10 (100%)** |
| Adversarial correct-abstention rate (`data/gold/adversarial_qa.jsonl`, 5 questions) | **4/5 (80%)** |

Reproduce: `python -m argus.rag.eval`

**Known failure mode:** the question *"What is the current repo rate set by the RBI
Monetary Policy Committee?"* false-positives against `NGFS-SCENARIO-GUIDANCE::Scenario_Families`
(score 0.286) because that chunk's text repeats the word "policy" three times ("policy
action" x3) and the query contains "Monetary Policy Committee" — a real limitation of
lexical (TF-IDF) retrieval on a small corpus: it matches on shared vocabulary, not
meaning. Two documented mitigations, not yet implemented:

1. Grow the regulatory corpus (more real RBI/BIS/NGFS documents indexed — the planned
   Month 3→ongoing task) so vocabulary overlap becomes less discriminative on its own.
2. Swap in `rag/embeddings.py`'s `SemanticRetriever` (the `semantic-rag` extra) which
   scores on meaning rather than shared terms — same `retrieve()` interface, no other
   module changes.

This is tracked, not hidden: `tests/adversarial/test_abstention_and_grounding.py::test_rag_agent_abstains_on_out_of_corpus_question`
encodes the current tolerance (a grounded answer to that specific question must at
least carry a low confidence score) rather than silently asserting a false 100%.

## Numeric-consistency & citation-grounding guarantees

Enforced programmatically in `agents/narrative_agent.py`, not just prompted for:

- Every drafted paragraph's composite hazard score is checked against the scoring
  engine's actual output (`_verify_numeric_consistency`) — a mismatch raises
  `GroundingError` before the item reaches the reviewer queue.
- Every citation is checked against the chunk the RAG agent actually retrieved
  (`_verify_citation`) — an LLM backend can never assert an uncited-but-plausible
  clause.

`tests/adversarial/test_abstention_and_grounding.py` exercises both guarantees
directly, including the case of a deliberately fabricated citation.

### Bug found and fixed: citation differentiation by district risk status

Found during a manual enterprise-level QA pass over the actual running Streamlit
console (inspecting the Human Review Queue and Audit Log tabs directly, not just
re-running the automated suite) — a reminder that reading real output, not just
green test runs, is what caught this one.

**Symptom:** every district's drafted narrative — high-risk or not — cited the
identical regulatory clause, `[RBI-2024-CLIMATE-DISC §Metrics and Targets]`. A
high-risk district (Cuttack, Alappuzha, Puri) should instead cite
`BIS-2024-CLIMATE-DISC §Physical Risk Disclosure`, per the intended business rule.

**Root cause:** `agents/rag_agent.py`'s `citation_for_district()` picked the clause
via a TF-IDF query ("find the passage about physical risk exposure by geography")
rather than a direct lookup. On this build's small 12-chunk corpus, both the
high-risk and low-risk queries top-ranked the same chunk
(`RBI-2024-CLIMATE-DISC::Metrics_and_Targets`, scoring 0.210 and 0.205
respectively) because that chunk shares enough vocabulary ("physical risk",
"geography", "disclose") to outrank the intended BIS chunk, which only scored
0.111 for the high-risk query. The RAG retrieval eval above never caught this
because it evaluates free-text regulatory Q&A (`rag_agent.ask()`), not this
separate district-citation code path.

**Fix:** added `TfidfRetriever.get(doc_id, section)` — a deterministic exact-match
lookup, scored `1.0` since it isn't a similarity match — and rewrote
`citation_for_district()` to call it directly instead of querying. Which clause
applies to a high-risk vs. within-tolerance district is a known, fixed business
rule, not an open retrieval question, so a lookup is the correct tool, not TF-IDF
ranking. A defensive fallback to the old query-based approach remains in place
only in case the corpus is ever edited and a section is renamed or removed.

**Regression test:** `tests/adversarial/test_abstention_and_grounding.py::test_high_risk_and_low_risk_districts_cite_different_clauses`
pins the fix by asserting the two citations resolve to different chunk IDs and the
expected doc/section pair. Re-verified end-to-end via
`python -m argus.agents.orchestrator`: Cuttack, Alappuzha and Puri (the three
HIGH RISK districts) now cite `BIS-2024-CLIMATE-DISC §Physical Risk Disclosure`;
the other five districts cite `RBI-2024-CLIMATE-DISC §Metrics and Targets`.

### Bug found and fixed: fresh-install toolchain drift masked real lint/type issues

Found during a reproducibility re-check of the delivered zip: this development
sandbox has a stray, globally-installed `ruff` (0.15.11, `uv tool install`) and
`mypy` (1.20.2) sitting ahead of the project's own `dev` extra on `PATH`, so every
`ruff check .` / `mypy src/argus` command run during earlier development quietly
used those older global tools rather than the versions `pip install -e ".[dev]"`
actually resolves (`ruff` 0.16.7, `mypy` 2.3.1 as of this check) — and reported
clean. Extracting the zip fresh into an isolated venv and running its *own*
installed `ruff`/`mypy` surfaced 15 real lint findings and 2 real type errors that
the sandbox's stale global tools had never seen: unsorted imports, a `dict()` call
better written as a literal, three `try/except Exception: pass` blocks that
silently discarded live-fetch failures with no log trail, two `pytest.raises(Exception)`
assertions broad enough to also pass on an unrelated bug, two stale `# noqa: F401`
comments on imports that were actually used, and two numpy typing errors under the
newer numpy/mypy pairing (`np.rot90`'s `k` argument needed an explicit `int()`
cast off a `rng.integers()` scalar; a `green`/`nir` tile pair needed both narrowed
together, not just `green`, for mypy to accept the follow-on call). All are fixed
in this build (see the three `ingestion/*.py` modules, `app/streamlit_app.py`,
`tests/data_quality/test_schemas.py`, `tests/integration/test_llm_backends.py`,
`rag/embeddings.py`, `augmentation/diffusion_augment.py`, `geospatial/hazard_scoring.py`)
and re-verified clean against a fully fresh `pip install -e ".[dev]"` in an
isolated venv — not against this sandbox's ambient tools. Recorded here rather
than quietly fixed and left undocumented, since "ruff/mypy clean" is a claim this
README repeats and it needs to mean what it says for anyone who actually clones
the repo.

### Bug found and fixed: unpinned `mcp` dependency resolved to a breaking major version

Also found during that same fresh-install re-check: `pyproject.toml` declared
`mcp>=1.6` with no upper bound. A fresh `pip install -e ".[dev]"` today resolves
`mcp` 2.x, which renamed `FastMCP` to `MCPServer` and changed the tool-registration
API — `mcp_server/server.py` is written against the 1.x `FastMCP` decorator API, so
a clean install broke at `tests/integration/test_mcp_tools.py`'s collection step
with `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`. This build's own
editable install (done earlier in this session, before mcp 2.x's release date)
never re-resolved the dependency and so never hit this. Fixed by pinning
`mcp>=1.6,<2` in `pyproject.toml`; upgrading to the 2.x API is a deliberate
follow-on task, not a drop-in bump. Re-verified: a fresh extraction + fresh venv
install now resolves `mcp` 1.30.0 and the full suite collects and passes.

## Bugs found and fixed during the Enterprise rewrite

### Unbounded hazard-contribution summation could exceed 100% of exposure

The first draft of `financial/impact_engine.py` summed `(hazard_score/100) *
vulnerability_weight * sector_exposure` independently across all four hazards, with
no shared cap. An integration-style assertion
(`physical_impact_inr_cr <= total_exposure_inr_cr`) caught this immediately: a
sector highly vulnerable to several severe hazards at once could show combined
"impact" exceeding 100% of its own exposure (e.g. Kanchipuram's
`climate_exposed_exposure_inr_cr` came out at ₹53,831.8 Cr against a
`public_exposure_inr_cr` of only ₹51,982.2 Cr) — nonsensical to a risk officer
reading the number. **Fix:** fold `HAZARD_WEIGHTS` (which sum to 1.0) into each
hazard's term (`sensitivity_share = HAZARD_WEIGHTS[hazard] * (score/100) *
vulnerability_weight`), which guarantees `combined_sensitivity ∈ [0, 1]` and
therefore `physical_impact_inr_cr <= sector_exposure_inr_cr` always. Applied
identically in `stress_testing.py`. Documented directly in `impact_engine.py`'s own
module docstring, naming that an earlier draft had the bug. Regression tests:
`tests/unit/test_exposure_engine.py::test_climate_exposed_exposure_never_exceeds_public_exposure`,
`tests/unit/test_stress_testing.py::test_stressed_exposure_never_exceeds_the_affected_exposure_base`.

### Two real IMD district-mapping gaps in the heat/temperature ingestion

`ingestion/heat_imd.py`'s district-to-station lookup initially missed two of the
eight demo districts entirely:

- **Kamrup** — the real IMD station climatology dataset labels this district's
  station `"KAMRUP (RURAL)"`, not `"KAMRUP"` (the district was administratively
  split after the station network was set up). Fixed with an explicit
  `DISTRICT_LABEL_ALIAS` mapping, kept as `status=Observed` (it *is* Kamrup's own
  station, just under an aliased label).
- **Kanchipuram** — has no IMD climatology station of its own in this dataset at
  all. Fixed with an explicit `DISTRICT_TO_STATION_OVERRIDE` to the nearest real
  station (Chennai, ~70km away), explicitly returned as `status=Proxy` so it is
  never silently presented as Kanchipuram's own reading. Regression test:
  `tests/unit/test_ingestion_fallbacks.py::test_heat_imd_labels_the_proxy_district_correctly`.

### Financial Translation's two headline figures are not "before vs. after"

Writing `tests/unit/test_financial_translation.py`, an initial assumption
("Severe Stress exposure should always be >= Climate-exposed exposure for the same
district") turned out to be false for 2 of the 8 demo districts (Kamrup, Puri) —
not a bug, but a real interaction between the two figures' different exposure
bases (full exposure at an unstressed hazard score, vs. 75%-of-exposure at a
capped, stressed score). See `financial/translation.py`'s module docstring for the
full explanation and why a production UI should say so explicitly rather than let a
reader assume monotonicity — the Streamlit Financial Translation tab now carries
this caveat directly under the table.

## Hazard-score backtesting

Not yet done against real historical loss data — this is the Month 2 task the roadmap
describes (backtest the composite score against 2–3 known events, e.g. the 2018 Kerala
floods) once real NDMA/insurance loss figures are sourced. The current demo sample's
district hazard profile is directionally consistent with public knowledge (Alappuzha
and Kozhikode, Kerala — real flood-prone districts — score highest on the flood
component; Latur and Banda — real drought-affected districts — score highest on
drought), which is a sanity check, not a backtest.

## Security

`pip-audit` (see `.github/workflows/ci.yml`) currently surfaces outdated *transitive*
dependencies pulled in by `streamlit`/`mcp` (e.g. `starlette`, `python-multipart`,
`pypdf`) and a few base-image packages (`setuptools`, `wheel`) — none in code this
project calls directly, none currently blocking CI (`|| true`), all worth a routine
`pip install -U` pass as a Month 6 hardening task rather than an emergency. No PII or
account-level financial data is in scope anywhere in this project, by design.
