import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Protocol

import httpx

from app.core.http_retry import post_with_retry
from app.exceptions.exception import RerankError
from app.retrieval.protocol import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]: ...


class JinaReranker:
    def __init__(
        self,
        client: httpx.Client,
        model: str,
        max_retries: int,
        backoff_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        response = post_with_retry(
            self._client,
            "/rerank",
            {
                "model": self._model,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
                "return_documents": False,
            },
            label="rerank",
            fail=RerankError,
            max_retries=self._max_retries,
            backoff_seconds=self._backoff_seconds,
            sleep=self._sleep,
        )
        try:
            results = response.json()["results"]
            ranked = [(int(item["index"]), float(item["relevance_score"])) for item in results]
        except (ValueError, KeyError, TypeError) as error:
            raise RerankError("rerank response was not understood") from error
        if any(not 0 <= index < len(documents) for index, _ in ranked):
            raise RerankError("rerank response referenced an unknown document")
        return ranked


class RerankingRetriever:
    def __init__(self, inner: Retriever, reranker: Reranker, top_n: int) -> None:
        self._inner = inner
        self._reranker = reranker
        self._top_n = top_n

    def retrieve(
        self, query: str, document_ids: Sequence[int] = ()
    ) -> list[RetrievedChunk]:
        candidates = self._inner.retrieve(query, document_ids)
        if len(candidates) <= 1:
            return candidates[: self._top_n]
        try:
            ranked = self._reranker.rerank(
                query, [chunk.content for chunk in candidates], self._top_n
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("rerank failed, keeping retrieval order: %s", error)
            return candidates[: self._top_n]
        return [replace(candidates[index], rerank_score=score) for index, score in ranked][
            : self._top_n
        ]
