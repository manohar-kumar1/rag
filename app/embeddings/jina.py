import time
from collections.abc import Callable, Sequence

import httpx

from app.embeddings.errors import EmbeddingDimensionMismatch, EmbeddingError

PASSAGE_TASK = "retrieval.passage"
QUERY_TASK = "retrieval.query"
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class JinaEmbedder:
    def __init__(
        self,
        client: httpx.Client,
        model: str,
        dimension: int,
        batch_size: int,
        max_retries: int,
        backoff_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, PASSAGE_TASK)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], QUERY_TASK)[0]

    def _embed(self, texts: Sequence[str], task: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch, task))
        return vectors

    def _embed_batch(self, batch: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": batch,
            "task": task,
            "dimensions": self._dimension,
        }
        items = sorted(self._post(payload)["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"jina returned {len(vectors)} embeddings for {len(batch)} inputs"
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingDimensionMismatch(self._dimension, len(vector))
        return vectors

    def _post(self, payload: dict) -> dict:
        for attempt in range(self._max_retries + 1):
            is_last = attempt == self._max_retries
            try:
                response = self._client.post("/embeddings", json=payload)
            except httpx.TransportError as error:
                if is_last:
                    raise EmbeddingError(
                        f"jina request failed: {type(error).__name__}"
                    ) from error
            else:
                if response.status_code == httpx.codes.OK:
                    return response.json()
                retryable = response.status_code in RETRYABLE_STATUSES
                if is_last or not retryable:
                    raise EmbeddingError(
                        f"jina request failed with status {response.status_code}"
                    )
            self._sleep(self._backoff_seconds * 2**attempt)
        raise EmbeddingError("jina request failed")
