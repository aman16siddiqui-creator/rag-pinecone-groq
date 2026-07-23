from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List

from modules.pdf_loader import PageContent

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover
    _HAS_LANGCHAIN = False


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_name: str
    page_number: int
    used_ocr: bool = False


def _make_chunk_id(doc_name: str, page_number: int, index: int, text: str) -> str:
    """Deterministic, collision-resistant chunk id so re-uploading the
    same document produces the same vector IDs (idempotent upserts)."""
    h = hashlib.sha1(f"{doc_name}|{page_number}|{index}|{text[:50]}".encode()).hexdigest()[:12]
    return f"{doc_name}_p{page_number}_c{index}_{h}"


def _simple_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fallback splitter: fixed-size sliding window with overlap,
    snapped to the nearest sentence-ending punctuation where possible."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to end on a sentence boundary within a small look-ahead window
        window = text[end: end + 80]
        period_idx = window.find(". ")
        if period_idx != -1 and end < n:
            end = end + period_idx + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_pages(
    pages: List[PageContent],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """Turn a list of per-page text into a flat list of Chunk objects,
    each tagged with its source document name and originating page
    number (required for source attribution later)."""

    all_chunks: List[Chunk] = []

    if _HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    for page in pages:
        if not page.text or not page.text.strip():
            continue

        if _HAS_LANGCHAIN:
            pieces = splitter.split_text(page.text)
        else:
            pieces = _simple_split(page.text, chunk_size, chunk_overlap)

        for idx, piece in enumerate(pieces):
            if len(piece.strip()) < 10:
                continue  # skip near-empty fragments
            all_chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(page.source_doc, page.page_number, idx, piece),
                    text=piece.strip(),
                    doc_name=page.source_doc,
                    page_number=page.page_number,
                    used_ocr=page.used_ocr,
                )
            )

    return all_chunks
