"""Citation-grounded retrieval over the regulatory corpus.

Uses TF-IDF + cosine similarity (scikit-learn) rather than a transformer embedding
model as the V1 default: it needs no multi-GB model download, indexes and queries in
milliseconds on a CPU-only laptop, and for a corpus this size and this domain-specific
(regulatory clause retrieval, not open-domain semantic search) gives retrieval quality
that's genuinely competitive with a small sentence-transformer — while meeting the
old-laptop constraint more strictly. ``semantic-rag`` in pyproject.toml documents the
drop-in sentence-transformers + ChromaDB upgrade path (same interface, swap
``TfidfRetriever`` for a ``SemanticRetriever`` implementing the same ``retrieve()``
signature) once heavier install budget is available.

Every retrieval below its minimum-score threshold returns no result — the agent that
calls this must abstain rather than let the LLM invent a citation. This is what makes
the hallucination/grounding tests in tests/adversarial/ meaningful.

Certification applied: NVIDIA — Intro to Transformer-Based NLP.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from argus.common.config import SETTINGS
from argus.rag.chunking import Chunk, load_corpus_chunks


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = chunks if chunks is not None else load_corpus_chunks()
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, *, top_k: int | None = None, min_score: float | None = None) -> list[RetrievalResult]:
        top_k = top_k or SETTINGS.retrieval_top_k
        min_score = SETTINGS.retrieval_min_score if min_score is None else min_score

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda cs: cs[1], reverse=True)

        results = [RetrievalResult(chunk=c, score=float(s)) for c, s in ranked[:top_k] if s >= min_score]
        return results

    def best(self, query: str, *, min_score: float | None = None) -> RetrievalResult | None:
        results = self.retrieve(query, top_k=1, min_score=min_score)
        return results[0] if results else None

    def get(self, doc_id: str, section: str) -> RetrievalResult | None:
        """Direct, deterministic lookup by (doc_id, section) — a score of 1.0 since
        this isn't a similarity match, it's an exact citation. Used wherever the
        correct clause is a known, fixed business rule (see agents/rag_agent.py's
        ``citation_for_district``) rather than an open question that genuinely needs
        semantic search. TF-IDF ranking is the wrong tool for "pick clause X for case
        Y" — that's a lookup, and pretending it's a retrieval problem is how a small
        corpus's lexical overlap silently picks the wrong clause (see
        docs/evaluation-report.md's "citation differentiation" note)."""
        for chunk in self.chunks:
            if chunk.doc_id == doc_id and chunk.section == section:
                return RetrievalResult(chunk=chunk, score=1.0)
        return None


if __name__ == "__main__":
    retriever = TfidfRetriever()
    for q in [
        "What should the Board disclose about climate risk oversight?",
        "What is the capital of France?",
    ]:
        best = retriever.best(q)
        print(f"Q: {q}")
        print(f"  -> {best.chunk.chunk_id} (score={best.score:.3f})" if best else "  -> ABSTAIN (no confident match)")
