"""Argus — Climate Physical-Risk Enterprise Console.

A real Streamlit dashboard wired to the actual pipeline — every number on screen
comes from geospatial/hazard_scoring.py, financial/impact_engine.py,
financial/stress_testing.py, analytics/concentration.py, rag/retriever.py and
common/audit.py, computed live when the page loads. Nothing here is a mockup.

DETERMINISTIC ENGINE vs LLM — the boundary this whole console is built around
(item #10 of the Enterprise spec, docs/architecture.md has the full write-up):

    Hazard data (IMD)  -->  DETERMINISTIC ENGINE  -->  RESULTS  -->  LLM
    Exposure data           (hazard_scoring,            (scores,       |
    Vulnerability model      impact_engine,              exposure,     v
                              stress_testing,             impact,   Explanation
                              concentration)              stress)    only

Every number in every tab below is produced by a plain, auditable Python function
in src/argus -- no LLM call is ever in the path that computes a score, an exposure
figure, or an impact estimate. The Narrative Agent (Review Queue tab) is the ONE
place an LLM is used to generate prose, and even there it is only allowed to
restate numbers the deterministic engine already computed (narrative_agent.py's
``_verify_numeric_consistency`` rejects a draft that invents or alters a figure).
The Regulatory Intelligence tab's RAG agent is the other LLM-adjacent surface, and
it is retrieval-only: it can quote and cite an indexed regulatory passage, or
abstain, never compute a risk number.

Run it:

    streamlit run app/streamlit_app.py

Certification applied: Mastercard — Advisors & Consulting Services Job Simulation
(stakeholder-facing framing); Google for Startups — Prompt to Prototype (this is the
month-5 MVP surface the roadmap builds outward from).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from argus.agents import rag_agent, scoring_agent
from argus.agents.orchestrator import run as run_pipeline
from argus.agents.reviewer_gate import CALCULATION_VERSION, MODEL_VERSION, approve, publish, reject
from argus.analytics.concentration import (
    concentration_by_hazard,
    concentration_by_sector,
    concentration_by_state,
    concentration_by_state_hazard,
    sector_geography_insight,
    top_concentration_insight,
)
from argus.common.audit import AuditLog
from argus.common.config import DATA_VERSION, HAZARD_WEIGHTS, HIGH_RISK_THRESHOLD, STRESS_SCENARIOS
from argus.explainability.feature_contributions import explain
from argus.financial.impact_engine import compute_physical_impact, rollup_by_district
from argus.financial.stress_testing import (
    district_scenario_summary,
    run_all_scenarios,
)
from argus.financial.translation import build_financial_translation
from argus.financial.vulnerability import sector_profile
from argus.geospatial.asset_exposure import (
    asset_locations_provenance,
    compute_asset_level_exposure,
    load_demo_asset_locations,
)

st.set_page_config(page_title="Argus — Climate Risk Enterprise Console", layout="wide")
st.title("Argus — Climate Physical-Risk Enterprise Console")
st.caption(
    "Hazard components are built from real public data (IMD cyclone best-track, IMD "
    "sub-divisional rainfall, IMD station climatology). Financial exposure is a "
    "PUBLIC-DATA EXPOSURE ESTIMATE (NABARD state credit potential apportioned by "
    "Census 2011 district population share) — not any specific bank's loan book. "
    "See the Data Lineage tab for full provenance, and docs/data-sources.md."
)

audit = AuditLog()


@st.cache_data(show_spinner="Running the deterministic Hazard → Exposure → Vulnerability pipeline...")
def load_pipeline_output():
    return scoring_agent.run_full_pipeline()


output = load_pipeline_output()
exposure_df = output.exposure_at_risk
asset_df = compute_asset_level_exposure(load_demo_asset_locations(), output.hazard_scores)
financial_translation_df = build_financial_translation(output.hazard_scores, output.sector_exposure, output.public_exposure)

(
    tab_map,
    tab_exposure,
    tab_asset,
    tab_translation,
    tab_concentration,
    tab_stress,
    tab_whatif,
    tab_reg,
    tab_lineage,
    tab_review,
    tab_decision,
) = st.tabs(
    [
        "Overview & Hazard Map",
        "Exposure & Vulnerability",
        "Asset-Level Intelligence",
        "Financial Translation",
        "Concentration Analysis",
        "Stress Testing",
        "What-If Analysis",
        "Regulatory Intelligence",
        "Data Lineage",
        "Review Queue",
        "Decision Workflow & Audit Log",
    ]
)

# ---------------------------------------------------------------------------
# Overview & Hazard Map
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader(f"District composite hazard score (high-risk threshold: {HIGH_RISK_THRESHOLD:.0f}/100)")
    coords = output.districts[["district", "lat", "lon"]]
    plot_df = exposure_df.merge(coords, on="district", how="left")
    fig = px.scatter_map(
        plot_df,
        lat="lat",
        lon="lon",
        size="climate_exposed_exposure_inr_cr",
        color="composite_hazard_score",
        color_continuous_scale="YlOrRd",
        hover_name="district",
        hover_data={
            "composite_hazard_score": ":.1f",
            "climate_exposed_exposure_inr_cr": ":,.0f",
            "primary_hazard": True,
            "most_affected_sector": True,
            "lat": False,
            "lon": False,
        },
        zoom=3.6,
        center={"lat": 20.5, "lon": 80.0},
        height=520,
        map_style="open-street-map",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, width="stretch")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts covered", len(exposure_df))
    c2.metric("High-risk districts", int(exposure_df["is_high_risk"].sum()))
    c3.metric("Public-data exposure (all districts)", f"₹{exposure_df['public_exposure_inr_cr'].sum():,.0f} Cr")
    c4.metric("Climate-exposed exposure (all districts)", f"₹{exposure_df['climate_exposed_exposure_inr_cr'].sum():,.0f} Cr")
    st.info(
        "**Where is the portfolio financially vulnerable?** The map above answers that at a glance — "
        "bubble size is climate-exposed exposure, colour is hazard severity. See the Concentration "
        "Analysis tab for which state/sector/hazard combinations actually drive that exposure."
    )

# ---------------------------------------------------------------------------
# Exposure & Vulnerability
# ---------------------------------------------------------------------------
with tab_exposure:
    st.subheader("Climate-Adjusted Exposure at Risk by district")
    st.caption(
        "climate_exposed_exposure_inr_cr = Hazard × Exposure × Vulnerability, summed across sectors "
        "(financial/impact_engine.py) — not a flat percentage of public_exposure_inr_cr."
    )
    st.dataframe(
        exposure_df.style.format(
            {
                "composite_hazard_score": "{:.1f}",
                "flood_component": "{:.1f}",
                "drought_component": "{:.1f}",
                "cyclone_component": "{:.1f}",
                "heat_component": "{:.1f}",
                "public_exposure_inr_cr": "₹{:,.0f} cr",
                "climate_exposed_exposure_inr_cr": "₹{:,.0f} cr",
            }
        ),
        width="stretch",
    )

    st.divider()
    st.subheader("Why did a district score the way it did?")
    chosen = st.selectbox("District", exposure_df["district"].tolist(), key="explain_district")
    row = exposure_df[exposure_df["district"] == chosen].iloc[0].to_dict()
    contributions = explain(row)
    contrib_fig = px.bar(
        x=[c.contribution for c in contributions],
        y=[c.hazard for c in contributions],
        orientation="h",
        labels={"x": "Contribution to composite score", "y": "Hazard"},
        title=f"{chosen} — composite {row['composite_hazard_score']:.1f}/100",
    )
    st.plotly_chart(contrib_fig, width="stretch")

    st.divider()
    st.subheader("Sector vulnerability profile")
    st.caption("How much of each hazard's severity converts into financial impact, by sector (financial/vulnerability.py).")
    profile_rows = [{"sector": s, **sector_profile(s)} for s in ("Agriculture", "MSME", "Other")]
    st.dataframe(pd.DataFrame(profile_rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Asset-Level Intelligence
# ---------------------------------------------------------------------------
with tab_asset:
    st.subheader("Asset/location-level physical climate impact")
    st.warning(
        "**Illustrative demo data.** Real, individually-geolocated borrower/asset data is confidential "
        "loan-book information and cannot be published in a public project — every row below is a "
        "synthetic, clearly-labeled illustrative asset placed near a real district centroid, showing the "
        "SHAPE a deploying institution's own book would take. See "
        "`geospatial/asset_exposure.py` for the swap-in interface (`load_asset_locations(path=...)`) a "
        "real institution uses to replace this file with its own real, geolocated book — every "
        "calculation below runs unchanged against it."
    )
    asset_coords = asset_df.merge(output.districts[["district"]].drop_duplicates(), on="district", how="left")
    asset_fig = px.scatter_map(
        asset_df,
        lat="lat",
        lon="lon",
        size="asset_exposure_inr_cr",
        color="asset_physical_impact_inr_cr",
        color_continuous_scale="OrRd",
        hover_name="asset_name",
        hover_data={
            "district": True,
            "sector": True,
            "asset_exposure_inr_cr": ":,.1f",
            "asset_physical_impact_inr_cr": ":,.2f",
            "primary_hazard": True,
            "lat": False,
            "lon": False,
        },
        zoom=3.6,
        center={"lat": 20.5, "lon": 80.0},
        height=480,
        map_style="open-street-map",
    )
    asset_fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(asset_fig, width="stretch")

    st.dataframe(
        asset_df[
            [
                "asset_id",
                "asset_name",
                "sector",
                "district",
                "state",
                "flood_component",
                "drought_component",
                "cyclone_component",
                "heat_component",
                "asset_exposure_inr_cr",
                "asset_physical_impact_inr_cr",
                "primary_hazard",
            ]
        ].style.format(
            {
                "flood_component": "{:.1f}",
                "drought_component": "{:.1f}",
                "cyclone_component": "{:.1f}",
                "heat_component": "{:.1f}",
                "asset_exposure_inr_cr": "₹{:.1f} cr",
                "asset_physical_impact_inr_cr": "₹{:.2f} cr",
            }
        ),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Financial Translation
# ---------------------------------------------------------------------------
with tab_translation:
    st.subheader("Financial-risk translation")
    st.caption(
        "Instead of a bare composite score, the numbers a risk committee actually needs — every figure "
        "traceable to one deterministic module (see the Data Lineage tab)."
    )
    st.caption(
        "Note: Climate-exposed exposure applies today's unstressed hazard score to the full sector exposure; "
        "Severe-stress exposure applies a capped, stressed hazard score to only 75% of that exposure (the "
        "Severe Stress scenario's exposure-affected share). For an already-high-hazard district the two can be "
        "close, or severe-stress can even read lower — they answer different questions, not 'before vs. after' "
        "on the same base. See financial/translation.py's docstring."
    )
    display = financial_translation_df.rename(
        columns={
            "portfolio_exposure_inr_cr": "Portfolio exposure (₹ Cr)",
            "climate_exposed_exposure_inr_cr": "Climate-exposed exposure (₹ Cr)",
            "severe_stress_exposure_inr_cr": "Severe-stress exposure (₹ Cr)",
            "primary_hazard": "Primary hazard",
            "most_affected_sector": "Most affected sector",
        }
    )
    display["Estimated impact range (₹ Cr)"] = financial_translation_df.apply(
        lambda r: f"{r['estimated_impact_low_inr_cr']:,.0f} – {r['estimated_impact_high_inr_cr']:,.0f}", axis=1
    )
    st.dataframe(
        display[
            [
                "district",
                "state",
                "Portfolio exposure (₹ Cr)",
                "Climate-exposed exposure (₹ Cr)",
                "Severe-stress exposure (₹ Cr)",
                "Estimated impact range (₹ Cr)",
                "Primary hazard",
                "Most affected sector",
            ]
        ].style.format(
            {
                "Portfolio exposure (₹ Cr)": "{:,.0f}",
                "Climate-exposed exposure (₹ Cr)": "{:,.0f}",
                "Severe-stress exposure (₹ Cr)": "{:,.0f}",
            }
        ),
        width="stretch",
    )
    st.caption("portfolio_exposure_source: " + financial_translation_df["portfolio_exposure_source"].iloc[0])

# ---------------------------------------------------------------------------
# Concentration Analysis
# ---------------------------------------------------------------------------
with tab_concentration:
    st.subheader("Concentration analysis")
    st.success(top_concentration_insight(output.impact_detail))
    st.info(sector_geography_insight(output.impact_detail))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("By state")
        st.dataframe(concentration_by_state(output.impact_detail), width="stretch", hide_index=True)
    with c2:
        st.caption("By hazard")
        st.dataframe(concentration_by_hazard(output.impact_detail), width="stretch", hide_index=True)
    with c3:
        st.caption("By sector")
        st.dataframe(concentration_by_sector(output.impact_detail), width="stretch", hide_index=True)

    st.divider()
    st.caption("State × hazard cross-cut (the rollup the insight sentence above is drawn from)")
    heat_df = concentration_by_state_hazard(output.impact_detail)
    heat_pivot = heat_df.pivot(index="state", columns="hazard", values="physical_impact_inr_cr").fillna(0.0)
    heat_fig = px.imshow(
        heat_pivot,
        labels={"color": "Physical impact (₹ Cr)"},
        color_continuous_scale="YlOrRd",
        text_auto=".0f",
        aspect="auto",
    )
    st.plotly_chart(heat_fig, width="stretch")

# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------
with tab_stress:
    st.subheader("Climate stress testing")
    st.warning(
        "**Model estimates, not actual or projected bank losses.** Each scenario applies a disclosed "
        "hazard multiplier and exposure-affected share (common/config.py's STRESS_SCENARIOS) — a "
        "modelling assumption, not a measured outcome."
    )
    stress_detail = run_all_scenarios(output.hazard_scores, output.sector_exposure)
    summary = district_scenario_summary(stress_detail)

    scenario_order = list(STRESS_SCENARIOS.keys())
    scenario_summary_by_name = {
        name: summary[summary["scenario"] == name]["stressed_financial_exposure_inr_cr"].sum() for name in scenario_order
    }
    cols = st.columns(len(scenario_order))
    for col, name in zip(cols, scenario_order):
        params = STRESS_SCENARIOS[name]
        col.metric(
            name,
            f"₹{scenario_summary_by_name[name]:,.0f} Cr",
            help=f"hazard ×{params['hazard_multiplier']}, {params['exposure_affected_share']:.0%} of sector exposure treated as affected",
        )

    scenario_fig = px.bar(
        summary,
        x="district",
        y="stressed_financial_exposure_inr_cr",
        color="scenario",
        barmode="group",
        category_orders={"scenario": scenario_order},
        labels={"stressed_financial_exposure_inr_cr": "Stressed financial exposure (₹ Cr)"},
    )
    st.plotly_chart(scenario_fig, width="stretch")

    st.dataframe(
        summary.style.format(
            {
                "stressed_financial_exposure_inr_cr": "₹{:,.0f} cr",
                "estimated_impact_low_inr_cr": "₹{:,.0f} cr",
                "estimated_impact_high_inr_cr": "₹{:,.0f} cr",
            }
        ),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# What-If Analysis
# ---------------------------------------------------------------------------
with tab_whatif:
    st.subheader("What-if analysis")
    st.caption(
        "Interactive severity sliders re-run the SAME deterministic formulas "
        "(financial/impact_engine.py) against hypothetical hazard severities — the calculations "
        "remain deterministic; nothing here is an LLM guess."
    )
    s1, s2, s3, s4 = st.columns(4)
    flood_mult = s1.slider("Flood severity ×", 0.5, 3.0, 1.0, 0.1)
    cyclone_mult = s2.slider("Cyclone severity ×", 0.5, 3.0, 1.0, 0.1)
    drought_mult = s3.slider("Drought severity ×", 0.5, 3.0, 1.0, 0.1)
    heat_mult = s4.slider("Heat severity ×", 0.5, 3.0, 1.0, 0.1)

    what_if_hz = output.hazard_scores.copy()
    what_if_hz["flood_component"] = (what_if_hz["flood_component"] * flood_mult).clip(upper=100.0)
    what_if_hz["cyclone_component"] = (what_if_hz["cyclone_component"] * cyclone_mult).clip(upper=100.0)
    what_if_hz["drought_component"] = (what_if_hz["drought_component"] * drought_mult).clip(upper=100.0)
    what_if_hz["heat_component"] = (what_if_hz["heat_component"] * heat_mult).clip(upper=100.0)

    what_if_detail = compute_physical_impact(what_if_hz, output.sector_exposure)
    what_if_rollup = rollup_by_district(what_if_detail)

    baseline_total = output.impact_detail["physical_impact_inr_cr"].sum()
    what_if_total = what_if_detail["physical_impact_inr_cr"].sum()
    delta = what_if_total - baseline_total

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline climate-exposed impact", f"₹{baseline_total:,.0f} Cr")
    m2.metric("What-if climate-exposed impact", f"₹{what_if_total:,.0f} Cr", delta=f"{delta:,.0f} Cr")
    m3.metric("Geographic concentration insight", "updated below")
    st.info(top_concentration_insight(what_if_detail))

    st.caption("Impact by district under the chosen what-if severities, vs. baseline")
    compare = what_if_rollup[["district", "state", "physical_impact_inr_cr", "primary_hazard", "most_affected_sector"]].merge(
        output.impact_by_district[["district", "physical_impact_inr_cr"]].rename(
            columns={"physical_impact_inr_cr": "baseline_physical_impact_inr_cr"}
        ),
        on="district",
    )
    compare["delta_inr_cr"] = compare["physical_impact_inr_cr"] - compare["baseline_physical_impact_inr_cr"]
    st.dataframe(
        compare.sort_values("physical_impact_inr_cr", ascending=False).style.format(
            {
                "physical_impact_inr_cr": "₹{:,.0f} cr",
                "baseline_physical_impact_inr_cr": "₹{:,.0f} cr",
                "delta_inr_cr": "₹{:+,.0f} cr",
            }
        ),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Regulatory Intelligence
# ---------------------------------------------------------------------------
with tab_reg:
    st.subheader("Regulatory intelligence")
    st.caption(
        "Answers only from the indexed RBI/BIS/NGFS corpus — abstains rather than guessing "
        "(rag/retriever.py + agents/rag_agent.py). This is retrieval, not generation: the RAG agent "
        "never computes a risk number, only quotes and cites indexed regulatory text."
    )
    question = st.text_input(
        "Question", placeholder="What should entities disclose about Board oversight of climate risk?"
    )
    if question:
        result = rag_agent.ask_structured(question)
        if result.grounded:
            st.success(f"**Applicable requirement:** {result.applicable_requirement}")
        else:
            st.warning("No confident match found in the indexed corpus — routed for human review rather than guessed.")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Regulator:** {result.regulator}")
        c1.markdown(f"**Document:** {result.document}")
        c1.markdown(f"**Publication date:** {result.publication_date}")
        c2.markdown(f"**Section:** {result.section}")
        c2.markdown(f"**Confidence score:** {result.confidence_score:.3f}")
        c2.markdown(f"**Human review required:** {'Yes' if result.requires_human_review else 'No'}")
        if result.evidence:
            st.markdown("**Evidence:**")
            st.code(result.evidence, language=None)
        st.markdown(f"**Applicability:** {result.applicability}")
        if result.requires_human_review:
            st.warning("Flagged for human compliance review — confidence below the review floor or no match found.")

# ---------------------------------------------------------------------------
# Data Lineage
# ---------------------------------------------------------------------------
with tab_lineage:
    st.subheader("Data lineage")
    st.caption(
        "For every result on this console: what it is, what went into it, where each input came from, "
        "which model computed it, the exact formula, when it was generated, and its validation status."
    )
    lineage_district = st.selectbox("District", exposure_df["district"].tolist(), key="lineage_district")
    lrow = exposure_df[exposure_df["district"] == lineage_district].iloc[0].to_dict()

    st.markdown(f"### RESULT: {lineage_district} physical-risk score & climate-exposed exposure")
    r1, r2 = st.columns(2)
    r1.metric("Composite hazard score", f"{lrow['composite_hazard_score']:.1f} / 100")
    r2.metric("Climate-exposed exposure estimate", f"₹{lrow['climate_exposed_exposure_inr_cr']:,.0f} Cr")

    st.markdown("**INPUTS**")
    st.dataframe(
        pd.DataFrame(
            [
                {"hazard": "Flood", "component_score": lrow["flood_component"], "weight": HAZARD_WEIGHTS["flood"]},
                {"hazard": "Drought", "component_score": lrow["drought_component"], "weight": HAZARD_WEIGHTS["drought"]},
                {"hazard": "Cyclone", "component_score": lrow["cyclone_component"], "weight": HAZARD_WEIGHTS["cyclone"]},
                {"hazard": "Heat", "component_score": lrow["heat_component"], "weight": HAZARD_WEIGHTS["heat"]},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**DATA SOURCES**")
    prov_rows = []
    all_provenance = {**output.provenance, "asset_locations (Asset-Level Intelligence tab)": asset_locations_provenance()}
    for key, prov in all_provenance.items():
        d = prov.to_dict()
        prov_rows.append(
            {
                "source": key,
                "dataset": d["dataset"],
                "organization": d["source_organization"],
                "status": d["status"],
                "publication_date": d["publication_date"],
                "geographic_resolution": d["geographic_resolution"],
                "retrieved_at": d["retrieved_at"],
                "source_url": d["source_url"],
            }
        )
    st.dataframe(pd.DataFrame(prov_rows), width="stretch", hide_index=True)
    with st.expander("Full methodology text per source"):
        for key, prov in all_provenance.items():
            st.markdown(f"**{key}** ({prov.status.value})")
            st.caption(prov.methodology)

    st.markdown("**MODEL**")
    st.write(f"Calculation engine: `{CALCULATION_VERSION}`  ·  Narrative/explanation layer: `{MODEL_VERSION}`")
    st.caption(f"Data snapshot used: {DATA_VERSION}")

    st.markdown("**CALCULATION**")
    st.code(
        "composite_hazard_score = "
        f"{HAZARD_WEIGHTS['flood']}×flood + {HAZARD_WEIGHTS['drought']}×drought + "
        f"{HAZARD_WEIGHTS['cyclone']}×cyclone + {HAZARD_WEIGHTS['heat']}×heat\n\n"
        "climate_exposed_exposure_inr_cr = Σ over (sector, hazard) of:\n"
        "    HAZARD_WEIGHTS[hazard] × (hazard_component_score / 100) × vulnerability_weight(sector, hazard)\n"
        "    × sector_exposure_inr_cr",
        language=None,
    )

    st.markdown("**GENERATED**")
    st.write(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ") + " (this page render)")

    st.markdown("**STATUS**")
    matching_items = [i for i in audit.latest_status().items() if i[0].startswith(f"{lineage_district}::")]
    if matching_items:
        for item_id, status in matching_items:
            st.write(f"`{item_id}` — **{status}**")
    else:
        st.write("Not yet drafted in the current run — see the Review Queue tab to run the pipeline.")

# ---------------------------------------------------------------------------
# Review Queue
# ---------------------------------------------------------------------------
with tab_review:
    st.subheader("Human review queue")
    if st.button("Run pipeline (draft reports for every district)"):
        orchestrator_run = run_pipeline(audit=audit)
        st.session_state["latest_run_id"] = orchestrator_run.run_id
        st.success(
            f"Run `{orchestrator_run.run_id}` drafted {len(orchestrator_run.results)} report items — "
            "all pending review."
        )
        st.rerun()

    current_run_id = st.session_state.get("latest_run_id") or audit.latest_run_id()
    if current_run_id:
        st.caption(f"Showing the pending queue for run `{current_run_id}` (the most recent analysis run only).")
    pending = audit.pending_review(run_id=current_run_id)
    if not pending:
        st.info("No items pending review in the latest run. Click above to run the pipeline.")
    for item_id in pending:
        history = audit.history(item_id, run_id=current_run_id)
        latest = history[-1]
        with st.expander(f"{latest.district} — {item_id}"):
            st.write(latest.content)
            st.caption(f"Drafted by {latest.llm_backend}, citation: {latest.citation_doc_id or 'none'}")
            st.caption("Lifecycle so far: " + " → ".join(e.event for e in history))
            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve-{item_id}"):
                approve(item_id, latest.district, latest.content, reviewer="analyst", audit=audit, run_id=current_run_id)
                st.rerun()
            if col2.button("Reject", key=f"reject-{item_id}"):
                reject(item_id, latest.district, "Rejected from console", reviewer="analyst", audit=audit, run_id=current_run_id)
                st.rerun()

# ---------------------------------------------------------------------------
# Decision Workflow & Audit Log
# ---------------------------------------------------------------------------
with tab_decision:
    st.subheader("Decision workflow")
    st.caption("generated → evidence_validated → submitted_for_review → approved/edited/rejected → published")

    current_run_id = st.session_state.get("latest_run_id") or audit.latest_run_id()
    status_map = audit.latest_status(run_id=current_run_id)
    if status_map:
        status_df = pd.DataFrame([{"item_id": k, "status": v} for k, v in status_map.items()])
        st.dataframe(status_df, width="stretch", hide_index=True)

        approved_items = [item_id for item_id, s in status_map.items() if s == "approved"]
        if approved_items:
            st.markdown("**Publish an approved item**")
            to_publish = st.selectbox("Item", approved_items)
            if st.button("Publish"):
                hist = audit.history(to_publish, run_id=current_run_id)
                latest = hist[-1]
                publish(to_publish, latest.district, latest.content, reviewer="analyst", audit=audit, run_id=current_run_id)
                st.success(f"Published {to_publish}.")
                st.rerun()
    else:
        st.info("No decision-workflow items yet — run the pipeline from the Review Queue tab.")

    st.divider()
    st.subheader("Runs")
    runs = audit.runs()
    if runs:
        st.dataframe(pd.DataFrame([r.__dict__ for r in runs]), width="stretch", hide_index=True)
    st.caption(
        "Each analysis run gets its own run_id — the Review Queue and the status table above default to "
        "the LATEST run only, so repeated runs never make the audit trail look like duplicate/broken data "
        "(the fix for item #12: one analysis run → one run ID → one result set)."
    )

    st.divider()
    st.subheader("Immutable audit log (all runs)")
    run_filter = st.selectbox("Filter to run", ["All runs"] + [r.run_id for r in runs])
    events = audit.history(run_id=None if run_filter == "All runs" else run_filter)
    if events:
        st.dataframe(pd.DataFrame([e.__dict__ for e in events]), width="stretch")
    else:
        st.info("No audit events yet — run the pipeline from the Review Queue tab.")
