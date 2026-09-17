"""RAG evaluation: retrieval quality against the curated gold Q&A set.

Thresholds here are the actual measured numbers for the TF-IDF backend against the
bundled corpus (see docs/evaluation-report.md) — not aspirational targets. If the
corpus or retriever changes and a number moves, this test is meant to catch it and
docs/evaluation-report.md should be updated alongside the fix, not the other way
around.
"""

from argus.rag.eval import run_eval


def test_gold_recall_at_3_is_100_percent():
    report = run_eval(top_k=3)
    assert report.gold_recall_at_k == 1.0, report.failures


def test_adversarial_abstention_at_least_80_percent():
    report = run_eval(top_k=3)
    assert report.adversarial_abstention_rate >= 0.8, report.failures


def test_report_totals_match_fixture_sizes():
    report = run_eval()
    assert report.gold_total == 10
    assert report.adversarial_total == 5
