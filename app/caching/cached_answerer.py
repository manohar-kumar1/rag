import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace
from typing import Protocol

from app.answering.protocol import (
    Answer,
    Answerer,
    DoneEvent,
    Question,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from app.caching.answer_store import AnswerCache, CacheEntry
from app.caching.serialization import question_key
from app.embeddings.protocol import Embedder

logger = logging.getLogger(__name__)


class CorpusVersion(Protocol):
    def current(self) -> str: ...


def cacheable(answer: Answer) -> bool:
    return answer.answered and answer.grounded is not False


class CachingAnswerer:
    def __init__(
        self,
        inner: Answerer,
        cache: AnswerCache,
        embedder: Embedder,
        corpus: CorpusVersion,
        semantic_threshold: float,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._embedder = embedder
        self._corpus = corpus
        self._semantic_threshold = semantic_threshold

    async def ask(self, question: Question) -> Answer:
        if question.history or question.thread_id:
            return await self._inner.ask(question)
        entry = await self._entry(question)
        cached = await self._lookup(entry)
        if cached is not None:
            return cached
        embedding = await self._shadow_match(entry, question)
        answer = await self._inner.ask(question)
        await self._save(entry, question, embedding, answer)
        return answer

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        if question.history or question.thread_id:
            async with aclosing(self._inner.stream(question)) as events:
                async for event in events:
                    yield event
            return
        entry = await self._entry(question)
        cached = await self._lookup(entry)
        if cached is not None:
            yield SourcesEvent(cached.sources)
            yield TokenEvent(cached.text)
            yield DoneEvent(cached)
            return
        embedding = await self._shadow_match(entry, question)
        async with aclosing(self._inner.stream(question)) as events:
            async for event in events:
                if isinstance(event, DoneEvent):
                    await self._save(entry, question, embedding, event.answer)
                yield event

    async def _entry(self, question: Question) -> CacheEntry:
        version = await asyncio.to_thread(self._corpus.current)
        return CacheEntry(question.cache_scope, question_key(question.text), version)

    async def _lookup(self, entry: CacheEntry) -> Answer | None:
        try:
            answer = await asyncio.to_thread(self._cache.lookup, entry)
        except Exception as error:  # noqa: BLE001
            logger.warning("answer_cache=error lookup failed: %s", type(error).__name__)
            return None
        logger.info("answer_cache=%s scope=%s", "miss" if answer is None else "hit", entry.scope)
        return None if answer is None else replace(answer, cached=True)

    async def _shadow_match(self, entry: CacheEntry, question: Question) -> list[float] | None:
        try:
            embedding = await asyncio.to_thread(self._embedder.embed_query, question.text)
            match = await asyncio.to_thread(self._cache.nearest, entry, embedding)
        except Exception as error:  # noqa: BLE001
            logger.warning("answer_cache=error semantic check failed: %s", type(error).__name__)
            return None
        if match is not None and match.similarity >= self._semantic_threshold:
            logger.info(
                "answer_cache=shadow_hit similarity=%.3f scope=%s (not served)",
                match.similarity,
                entry.scope,
            )
        return embedding

    async def _save(
        self, entry: CacheEntry, question: Question, embedding: list[float] | None, answer: Answer
    ) -> None:
        if not cacheable(answer):
            return
        try:
            await asyncio.to_thread(self._cache.save, entry, question.text, embedding, answer)
        except Exception as error:  # noqa: BLE001
            logger.warning("answer_cache=error save failed: %s", type(error).__name__)
