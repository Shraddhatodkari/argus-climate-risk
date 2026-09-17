"""Semantic (sentence-transformer) embedding backend — the documented upgrade path
for retriever.py, kept out of the default install.

retriever.py's TfidfRetriever is the V1 default (see its module docstring for why: no
multi-GB model download, millisecond queries on a CPU-only laptop). This module is the
drop-in replacement once the ``semantic-rag`` extra is installed
(`pip install -e ".[semantic-rag]"`, pulls sentence-transformers + chromadb) — same
``retrieve()`` contract as TfidfRetriever, so agents/rag_agent.py does not change when
it's swapped in.

Certification applied: NVIDIA — Intro to Transformer-Based NLP.
"""

from __future__ import annotations

from argus.rag.chunking import Chunk, load_corpus_chunks
from argus.rag.retriever import RetrievalResult

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # CPU-friendly; ~130MB


class SemanticRetriever:
    """Same interface as TfidfRetriever (retrieve/best), backed by sentence-transformer
    embeddings persisted in a local ChromaDB collection. Requires the 'semantic-rag'
    extra — raises a clear, actionable error otherwise rather than a bare ImportError.
    """

    def __init__(self, chunks: list[Chunk] | None = None, *, persist_dir: str = "./data/processed/chroma") -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SemanticRetriever requires the 'semantic-rag' extra "
                '(`pip install -e ".[semantic-rag]"` — pulls sentence-transformers + '
                "chromadb). Use rag.retriever.TfidfRetriever for the zero-extra default."
            ) from exc

        self.chunks = chunks if chunks is not None else load_corpus_chunks()
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection("argus_regulatory_corpus")
        self._index()

    def _index(self) -> None:
        embeddings = self._model.encode([c.text for c in self.chunks]).tolist()
        self._collection.upsert(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=embeddings,
            documents=[c.text for c in self.chunks],
            metadatas=[{"doc_id": c.doc_id, "section": c.section, "source": c.source} for c in self.chunks],
        )

    def retrieve(self, query: str, *, top_k: int = 3, min_score: float = 0.35) -> list[RetrievalResult]:
        query_embedding = self._model.encode([query]).tolist()
        hits = self._collection.query(query_embeddings=query_embedding, n_results=top_k)
        results = []
        by_id = {c.chunk_id: c for c in self.chunks}
        for chunk_id, distance in zip(hits["ids"][0], hits["distances"][0]):
            similarity = 1.0 - distance  # cosine distance -> similarity
            if similarity >= min_score:
                results.append(RetrievalResult(chunk=by_id[chunk_id], score=similarity))
        return results

    def best(self, query: str, *, min_score: float = 0.35) -> RetrievalResult | None:
        results = self.retrieve(query, top_k=1, min_score=min_score)
        return results[0] if results else None


if __name__ == "__main__":
    retriever = SemanticRetriever()
    result = retriever.best("What should the Board disclose about climate risk oversight?")
    print(result)
