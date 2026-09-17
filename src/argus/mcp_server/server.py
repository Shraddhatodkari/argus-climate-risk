"""Argus MCP server — real tool definitions on top of the Anthropic MCP Python SDK's
FastMCP interface, calling directly into the same modules the Streamlit console and
the orchestrator use (no duplicated logic).

Run it:

    python -m argus.mcp_server.server

Then point any MCP client (Claude Desktop's config, Claude Code's `mcp add`, or a
custom client using the `mcp` SDK) at this process over stdio, and a risk analyst can
ask "what's the exposure-at-risk for Cuttack?" directly from their everyday LLM tool
instead of opening the dashboard.

Certification applied: Anthropic — Building with the Claude API.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from argus.agents import rag_agent, scoring_agent
from argus.explainability.feature_contributions import explain

mcp = FastMCP(
    "argus",
    instructions=(
        "Tools for Argus, a climate physical-risk intelligence platform for credit "
        "portfolios. Hazard components are built from real public data (IMD cyclone "
        "best-track, IMD sub-divisional rainfall, IMD station climatology); financial "
        "exposure figures are a PUBLIC-DATA EXPOSURE ESTIMATE (NABARD credit potential "
        "apportioned by Census 2011 population share), not any specific institution's "
        "loan book — see docs/data-sources.md for full provenance. A real bank's "
        "confidential portfolio can be substituted via ingestion.financial_exposure."
        "load_institution_portfolio without changing any downstream tool."
    ),
)


@mcp.tool()
def get_hazard_exposure(district: str) -> dict:
    """Return the composite climate hazard score, its flood/drought/cyclone/heat
    breakdown, and the public-data climate-exposed exposure estimate (in INR crore)
    for one district. Raises if the district isn't in the current dataset."""
    row = scoring_agent.run_for_district(district)
    contributions = explain(row)
    return {
        "district": row["district"],
        "state": row["state"],
        "composite_hazard_score": row["composite_hazard_score"],
        "primary_hazard": row["primary_hazard"],
        "most_affected_sector": row["most_affected_sector"],
        "is_high_risk": bool(row["is_high_risk"]),
        "public_exposure_inr_cr": row["public_exposure_inr_cr"],
        "climate_exposed_exposure_inr_cr": row["climate_exposed_exposure_inr_cr"],
        "exposure_basis": "Public-data exposure estimate (not a specific institution's portfolio)",
        "contributions": [c.__dict__ for c in contributions],
    }


@mcp.tool()
def list_high_risk_districts() -> list[dict]:
    """Return every district currently flagged high-risk, sorted by climate-exposed
    exposure (highest first)."""
    df = scoring_agent.run()
    high = df[df["is_high_risk"]].sort_values("climate_exposed_exposure_inr_cr", ascending=False)
    return high.to_dict(orient="records")


@mcp.tool()
def search_regulatory_corpus(question: str) -> dict:
    """Answer a climate-risk regulatory question with a citation into the indexed
    RBI/BIS/NGFS corpus, or explicitly abstain if nothing relevant is indexed. Never
    invents a citation that wasn't actually retrieved."""
    answer = rag_agent.ask(question)
    return {
        "question": answer.question,
        "grounded": answer.grounded,
        "answer": answer.answer_text,
        "source_doc_id": answer.citation.chunk.doc_id if answer.citation else None,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
