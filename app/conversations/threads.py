import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace
from typing import Protocol

from app.answering.protocol import (
    Answer,
    Answerer,
    DoneEvent,
    Question,
    StreamEvent,
    Turn,
)


class ConversationStore(Protocol):
    def recent(self, thread_id: str, limit: int) -> list[Turn]: ...

    def append(self, thread_id: str, turns: list[Turn]) -> None: ...


class ThreadedAnswerer:
    def __init__(self, inner: Answerer, store: ConversationStore, max_turns: int) -> None:
        self._inner = inner
        self._store = store
        self._max_turns = max_turns

    async def ask(self, question: Question) -> Answer:
        if question.thread_id is None:
            return await self._inner.ask(question)
        answer = await self._inner.ask(await self._with_history(question))
        await self._remember(question, answer)
        return replace(answer, thread_id=question.thread_id)

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        threaded = question if question.thread_id is None else await self._with_history(question)
        async with aclosing(self._inner.stream(threaded)) as events:
            async for event in events:
                if isinstance(event, DoneEvent) and question.thread_id is not None:
                    await self._remember(question, event.answer)
                    event = DoneEvent(replace(event.answer, thread_id=question.thread_id))
                yield event

    async def _with_history(self, question: Question) -> Question:
        history = await asyncio.to_thread(
            self._store.recent, question.thread_id, self._max_turns
        )
        return replace(question, history=tuple(history))

    async def _remember(self, question: Question, answer: Answer) -> None:
        turns = [Turn("user", question.text), Turn("assistant", answer.text)]
        await asyncio.to_thread(self._store.append, question.thread_id, turns)
