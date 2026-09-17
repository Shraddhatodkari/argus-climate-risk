"""Integration tests: MCP tool surface (called as plain Python callables — see
mcp_server/tools.py's docstring for why that's a faithful test of the MCP-exposed
behaviour without needing to spin up the stdio transport)."""

from argus.mcp_server.tools import (
    get_hazard_exposure,
    list_high_risk_districts,
    search_regulatory_corpus,
)


def test_get_hazard_exposure_returns_contributions_that_sum_to_the_score():
    result = get_hazard_exposure("Alappuzha")
    total = sum(c["contribution"] for c in result["contributions"])
    assert abs(total - result["composite_hazard_score"]) < 0.05


def test_list_high_risk_districts_only_returns_flagged_districts():
    high_risk = list_high_risk_districts()
    assert len(high_risk) > 0
    assert all(d["is_high_risk"] for d in high_risk)


def test_list_high_risk_districts_sorted_by_exposure_descending():
    high_risk = list_high_risk_districts()
    values = [d["climate_exposed_exposure_inr_cr"] for d in high_risk]
    assert values == sorted(values, reverse=True)


def test_search_regulatory_corpus_grounds_a_real_question():
    result = search_regulatory_corpus("What should entities disclose about Board oversight of climate risk?")
    assert result["grounded"] is True
    assert result["source_doc_id"] == "RBI-2024-CLIMATE-DISC"


def test_search_regulatory_corpus_abstains_on_fabricated_premise():
    result = search_regulatory_corpus("Which cryptocurrency does RBI recommend banks invest reserves in?")
    assert result["grounded"] is False
    assert result["source_doc_id"] is None
