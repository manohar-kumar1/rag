from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from app.embeddings.errors import EmbeddingDimensionMismatch


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str]) -> Any: ...

    def get_embedding_dimension(self) -> int | None: ...


class SentenceTransformerEmbedder:
    def __init__(self, encoder: SentenceEncoder, expected_dimension: int) -> None:
        actual = encoder.get_embedding_dimension()
        if actual != expected_dimension:
            raise EmbeddingDimensionMismatch(expected_dimension, actual)
        self._encoder = encoder

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return np.asarray(self._encoder.encode(list(texts)), dtype=float).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
