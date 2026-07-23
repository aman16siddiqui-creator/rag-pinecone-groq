from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        self._model = SentenceTransformer(self.model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of chunk texts (used during indexing)."""
        if not texts:
            return np.zeros((0, self.dimension))
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # so dot-product == cosine similarity
        )

    def embed_query(self, query: str) -> List[float]:
        """Embed a single user query (used at retrieval time)."""
        vector = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Cached singleton so the (relatively heavy) model is only loaded
    once per process — important inside a Streamlit app that reruns the
    script on every interaction."""
    return Embedder()
