"""Regulatory corpus loading and chunking.

The bundled corpus (data/regulatory_corpus/*.txt — paraphrased, clearly labeled
summaries of the real RBI/BIS/NGFS publications; see each file's header) uses a simple
[SOURCE: ...] / [DOC_ID: ...] / [SECTION: ...] markup so chunking can preserve exact
document + section provenance for every retrieved passage — the foundation the citation
checker in agents/narrative_agent.py relies on.

Certification applied: NVIDIA — Intro to Transformer-Based NLP (retrieval unit design).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from argus.common.config import CORPUS_DIR

DOC_ID_RE = re.compile(r"\[DOC_ID:\s*([^\]]+)\]")
SOURCE_RE = re.compile(r"\[SOURCE:\s*([^\]]+)\]")
SECTION_RE = re.compile(r"\[SECTION:\s*([^\]]+)\]")
REGULATOR_RE = re.compile(r"\[REGULATOR:\s*([^\]]+)\]")
PUBLISHED_RE = re.compile(r"\[PUBLISHED:\s*([^\]]+)\]")
DOC_URL_RE = re.compile(r"\[DOC_URL:\s*([^\]]+)\]")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    section: str
    text: str
    regulator: str = "Unknown"
    published: str = "Unknown"
    doc_url: str = ""


def _parse_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_id_match = DOC_ID_RE.search(raw)
    source_match = SOURCE_RE.search(raw)
    regulator_match = REGULATOR_RE.search(raw)
    published_match = PUBLISHED_RE.search(raw)
    doc_url_match = DOC_URL_RE.search(raw)
    doc_id = doc_id_match.group(1).strip() if doc_id_match else path.stem
    source = source_match.group(1).strip() if source_match else path.stem
    regulator = regulator_match.group(1).strip() if regulator_match else "Unknown"
    published = published_match.group(1).strip() if published_match else "Unknown"
    doc_url = doc_url_match.group(1).strip() if doc_url_match else ""

    sections = SECTION_RE.split(raw)
    # re.split on a capturing group returns [preamble, section_name, body, section_name, body, ...]
    chunks: list[Chunk] = []
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        # stop the body at the next bracketed marker if any slipped through
        body = re.split(r"\n\[", body)[0].strip()
        if not body:
            continue
        chunk_id = f"{doc_id}::{section_name}".replace(" ", "_")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                section=section_name,
                text=body,
                regulator=regulator,
                published=published,
                doc_url=doc_url,
            )
        )
    return chunks


def load_corpus_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        chunks.extend(_parse_document(path))
    if not chunks:
        raise RuntimeError(f"No regulatory corpus chunks found under {corpus_dir}.")
    return chunks


if __name__ == "__main__":
    for c in load_corpus_chunks():
        print(f"{c.chunk_id:55s} ({len(c.text)} chars)")
