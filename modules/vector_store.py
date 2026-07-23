from __future__ import annotations

from typing import Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from config import settings


class VectorStoreError(Exception):
    """Raised on any Pinecone connection / operation failure so the UI
    can show a friendly message instead of a raw traceback."""


class PineconeVectorStore:
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.api_key = api_key or settings.pinecone_api_key
        self.index_name = index_name or settings.pinecone_index_name
        self.dimension = dimension or settings.embedding_dimension

        if not self.api_key:
            raise VectorStoreError("PINECONE_API_KEY is missing.")

        try:
            self.pc = Pinecone(api_key=self.api_key)
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialise Pinecone client: {exc}") from exc

        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def _ensure_index(self):
        try:
            existing = [i["name"] for i in self.pc.list_indexes()]
            if self.index_name not in existing:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=settings.pinecone_metric,
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region,
                    ),
                )
        except PineconeException as exc:
            raise VectorStoreError(f"Pinecone index creation/check failed: {exc}") from exc
        except Exception as exc:
            raise VectorStoreError(f"Could not reach Pinecone: {exc}") from exc

    def delete_namespace(self, namespace: str):
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except PineconeException as exc:
            raise VectorStoreError(f"Failed to clear namespace '{namespace}': {exc}") from exc

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
        namespace: str,
        batch_size: int = 100,
    ):
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise VectorStoreError("ids, vectors, and metadatas must be the same length.")

        try:
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                batch = list(zip(ids[start:end], vectors[start:end], metadatas[start:end]))
                self.index.upsert(vectors=batch, namespace=namespace)
        except PineconeException as exc:
            raise VectorStoreError(f"Upsert to Pinecone failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        vector: List[float],
        top_k: int,
        namespace: str,
        metadata_filter: Optional[Dict] = None,
    ):
        try:
            result = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=metadata_filter or None,
            )
            return result.get("matches", []) if isinstance(result, dict) else result.matches
        except PineconeException as exc:
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc
        except Exception as exc:
            raise VectorStoreError(f"Unexpected error querying Pinecone: {exc}") from exc

    def describe_index_stats(self):
        try:
            return self.index.describe_index_stats()
        except Exception as exc:
            raise VectorStoreError(f"Could not fetch index stats: {exc}") from exc
