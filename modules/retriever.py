from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from modules.embedder import Embedder
from modules.vector_store import PineconeVectorStore


@dataclass
class RetrievedChunk:
    text: str
    doc_name: str
    page_number: int
    chunk_id: str
    score: float


class Retriever:
    def __init__(self, vector_store: PineconeVectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int,
        namespace: str,
        score_threshold: float = 0.0,
        page_filter: Optional[int] = None,
        doc_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        metadata_filter: Dict = {}
        if page_filter is not None:
            metadata_filter["page_number"] = {"$eq": page_filter}
        if doc_filter:
            metadata_filter["doc_name"] = {"$eq": doc_filter}

        query_vector = self.embedder.embed_query(query)
        matches = self.vector_store.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter or None,
        )

        def _to_chunk(m) -> RetrievedChunk:
            score = m["score"] if isinstance(m, dict) else m.score
            meta = m["metadata"] if isinstance(m, dict) else m.metadata
            match_id = m["id"] if isinstance(m, dict) else m.id
            return RetrievedChunk(
                text=meta.get("text", ""),
                doc_name=meta.get("doc_name", "unknown"),
                page_number=int(meta.get("page_number", -1)),
                chunk_id=match_id,
                score=float(score),
            )

        all_chunks = [_to_chunk(m) for m in matches]
        results = [c for c in all_chunks if c.score >= score_threshold]

        # Broad / whole-document questions ("what is this PDF about",
        # "summarize this document") often score lower on cosine similarity
        # than a narrow factual question, because there's no single
        # sentence matching the query wording. A hard threshold then wipes
        # out every chunk and the user gets a false "not available" even
        # though the document is right there. Fallback: if the strict
        # threshold produced nothing but Pinecone did return matches, use
        # the best few anyway (their true similarity scores still show).
        if not results and all_chunks:
            results = sorted(all_chunks, key=lambda c: c.score, reverse=True)[: min(3, len(all_chunks))]

        return results
