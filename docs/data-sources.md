# Data sources — Enterprise build

Every dataset below carries a full `DataProvenance` record (`common/provenance.py`) —
dataset name, source organization, source URL, publication date, observation period,
geographic resolution, unit, methodology, an Observed/Derived/Modelled/Proxy status,
and the UTC timestamp this build actually retrieved/computed it. The Data Lineage tab
in the Streamlit console reads these records directly; nothing below is asserted
without a citation back to them.

## Real data actually ingested in this build

| Source | Provides | Access | File / module | Status |
|---|---|---|---|---|
| IMD RSMC New Delhi — cyclone best-track | Every recorded cyclonic-storm fix in the North Indian Ocean, 1982–2026 (7,585 observations) | `imdtrack` PyPI package (live, monthly-refreshed), frozen local snapshot as offline fallback | `data/real/imd_cyclone_besttrack.csv`, `ingestion/cyclone_imd.py` | Observed (raw fixes) → Derived (per-district exposure score) |
| IMD sub-divisional monthly rainfall | 1901–2017 monthly/seasonal rainfall, 36 official IMD meteorological subdivisions (4,188 rows) | `raw.githubusercontent.com` snapshot of a public rainfall dataset | `data/real/imd_subdivision_rainfall_1901_2017.csv`, `ingestion/rainfall_imd.py` | Observed (raw totals) → Derived (SPI-like flood/drought proxy) |
| IMD station climatological normals | Monthly climatological statistics (mean/extreme daily max temperature, etc.) per IMD station (5,257 rows) | `raw.githubusercontent.com` snapshot of a public IMD-station climatology dataset | `data/real/imd_station_climatological_stats.csv`, `ingestion/heat_imd.py` | Observed (station normals) → Derived (per-district heat-exposure score); one district (Kanchipuram) uses the nearest available station and is explicitly returned as Proxy |
| NABARD State Credit Seminar / State Focus Paper — priority-sector credit potential | Real, individually-dated state-level priority-sector credit potential (6 states, each its own fiscal year and publication date) | Public NABARD/PIB announcements (retrieved via web search) | `data/real/nabard_state_credit_potential.csv`, `ingestion/financial_exposure.py` | Observed (state figure) → Derived (district apportionment) |
| NABARD sector composition | Representative national priority-sector split (Agriculture/MSME/Other) plus two state-specific splits (Maharashtra, Uttar Pradesh) | Public NABARD publications | `data/real/nabard_sector_composition.csv`, `financial/sector_allocation.py` | Derived (Observed where state-specific; Proxy where the representative national split stands in) |
| Census of India 2011 | State and district population | Public Census 2011 releases (retrieved via web search) | `data/real/census2011_population.csv`, `ingestion/financial_exposure.py` | Observed |
| RBI / BIS / NGFS climate-disclosure guidance | RAG grounding corpus, each document tagged with its real regulator, publication date, and source URL | Public documents, paraphrased into a small labeled corpus | `data/regulatory_corpus/*.txt`, `rag/chunking.py` | Derived (paraphrased summary of a real, cited publication) |

## Deliberately not real, and disclosed as such

| Source | Why it's a Proxy in this build | Module |
|---|---|---|
| Satellite NDWI flood-extent tile (0.6 weight inside `flood_component`) | This sandbox cannot reach a live Copernicus EMS/Sentinel-2 feed; a deterministic synthetic tile is generated instead, with a fixed `DataProvenance(status=PROXY)` record | `ingestion/copernicus.py`, `geospatial/hazard_scoring.SATELLITE_FLOOD_PROVENANCE` |
| Asset/borrower-level exposure locations | Real, individually-geolocated borrower data is confidential loan-book information and cannot be published in a public project; the bundled file is explicitly illustrative (`is_synthetic="true"` on every row, names prefixed "Illustrative...") | `data/samples/asset_locations.csv`, `geospatial/asset_exposure.py` |
| Institution portfolio exposure | No real bank/NBFC portfolio ships with this project — `load_institution_portfolio()` / `load_asset_locations()` return `None` / the demo file until a deploying institution configures its own | `ingestion/financial_exposure.py`, `geospatial/asset_exposure.py` |

**The public-data exposure estimate is not a bank's portfolio, and the console never
blurs the two.** `public_exposure_inr_cr` = NABARD's real, disclosed state priority-
sector credit potential apportioned to district level by real Census 2011 population
share — a genuine measure of a district's *addressable* climate-exposed lending
market, not any specific institution's outstanding loan book. Every place this number
appears (Financial Translation tab, MCP `get_hazard_exposure` tool, the Narrative
Agent's drafted text) says so explicitly. The architecture accepts a real institution
portfolio via `ingestion.financial_exposure.load_institution_portfolio(path=...)` and
`geospatial.asset_exposure.load_asset_locations(path=...)` without any downstream
module changing.

## What this sandbox environment actually reached vs. what production reaches

This development sandbox's network egress is restricted to an allowlist (PyPI,
npm, a handful of package registries). Three access patterns were used to still reach
genuinely real data within that constraint:

1. `pip install imdtrack` — PyPI is allowlisted, so the cyclone best-track package
   installs and its live-refresh path works exactly as it would in production.
2. `raw.githubusercontent.com` direct file fetches — reachable even though the GitHub
   web UI and API were not, once the exact file path was known (found via each
   repository's own README).
3. `WebSearch`/`WebFetch` (routed through separate infrastructure with broader reach)
   for the NABARD and Census figures, each individually verified and cited.

`mausam.imd.gov.in`, `rbi.org.in`, and WRI's own data endpoints were not reachable
from this sandbox during development — every ingestion module's live-fetch code path
is still real and documented in its own file (`ingestion/base.py`'s `safe_get`
degrades to the bundled snapshot on any network failure, never raises), so this is a
resilience feature exercised by an actually-restricted network, not untested logic.
`docs/evaluation-report.md` has the fuller retrieval notes, including two IMD
district-mapping issues found and fixed during this build (Kamrup's climatology
station is labeled "KAMRUP (RURAL)" in the source dataset; Kanchipuram has no station
of its own and falls back to Chennai, ~70km away, explicitly marked Proxy).
