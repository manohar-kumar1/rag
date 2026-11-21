import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import httpx

from app.exceptions.exception import RerankError
from app.retrieval.protocol import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]: ...


class JinaReranker:
    def __init__(self, client: httpx.Client, model: str) -> None:
        self._client = client
        self._model = model

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        try:
            response = self._client.post(
                "/rerank",
                json={
                    "model": self._model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": top_n,
                    "return_documents": False,
                },
            )
        except httpx.TransportError as error:
            raise RerankError(f"rerank request failed: {type(error).__name__}") from error
        if response.status_code != httpx.codes.OK:
            raise RerankError(f"rerank request failed with status {response.status_code}")
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

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        candidates = self._inner.retrieve(query)
        if len(candidates) <= 1:
            return candidates[: self._top_n]
        try:
            ranked = self._reranker.rerank(
                query, [chunk.content for chunk in candidates], self._top_n
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("rerank failed, keeping retrieval order: %s", type(error).__name__)
            return candidates[: self._top_n]
        return [replace(candidates[index], rerank_score=score) for index, score in ranked][
            : self._top_n
        ]
