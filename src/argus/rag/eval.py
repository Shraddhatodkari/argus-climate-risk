"""RAG evaluation harness.

Two real, programmatically-checkable measures — no LLM judge required, so this runs
in CI on every push:

  1. Doc-level recall@k against data/gold/rag_qa_gold.jsonl: does the retriever return
     at least one chunk from the expected source document in its top-k?
  2. Correct-abstention rate against data/gold/adversarial_qa.jsonl: does the retriever
     correctly return nothing for a question the corpus cannot answer?

Results here are written to docs/evaluation-report.md as the project progresses — see
that file for the current numbers and the one documented false-positive failure mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from argus.common.config import GOLD_DIR
from argus.rag.retriever import TfidfRetriever


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass(frozen=True)
class EvalReport:
    gold_recall_at_k: float
    gold_hits: int
    gold_total: int
    adversarial_abstention_rate: float
    adversarial_correct: int
    adversarial_total: int
    failures: list[str]


def run_eval(retriever: TfidfRetriever | None = None, *, top_k: int = 3) -> EvalReport:
    retriever = retriever or TfidfRetriever()
    gold = _load_jsonl(GOLD_DIR / "rag_qa_gold.jsonl")
    adversarial = _load_jsonl(GOLD_DIR / "adversarial_qa.jsonl")

    failures: list[str] = []

    gold_hits = 0
    for item in gold:
        results = retriever.retrieve(item["question"], top_k=top_k)
        doc_ids = {r.chunk.doc_id for r in results}
        if item["expected_doc_id"] in doc_ids:
            gold_hits += 1
        else:
            failures.append(f"MISS  gold: {item['question']!r} expected {item['expected_doc_id']}")

    adversarial_correct = 0
    for item in adversarial:
        results = retriever.retrieve(item["question"], top_k=1)
        if not results:
            adversarial_correct += 1
        else:
            failures.append(
                f"FALSE-POSITIVE  adversarial: {item['question']!r} -> "
                f"{results[0].chunk.chunk_id} (score={results[0].score:.3f})"
            )

    return EvalReport(
        gold_recall_at_k=gold_hits / len(gold),
        gold_hits=gold_hits,
        gold_total=len(gold),
        adversarial_abstention_rate=adversarial_correct / len(adversarial),
        adversarial_correct=adversarial_correct,
        adversarial_total=len(adversarial),
        failures=failures,
    )


if __name__ == "__main__":
    report = run_eval()
    print(f"Gold recall@3:            {report.gold_hits}/{report.gold_total} ({report.gold_recall_at_k:.0%})")
    print(
        f"Adversarial abstention:  {report.adversarial_correct}/{report.adversarial_total} "
        f"({report.adversarial_abstention_rate:.0%})"
    )
    for f in report.failures:
        print(f"  {f}")
