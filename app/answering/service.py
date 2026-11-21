import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from app.answering.citations import SentinelGate, cite, is_no_answer
from app.answering.generator import AnswerGenerator
from app.answering.prompts import NO_ANSWER
from app.answering.protocol import (
    NOT_FOUND_MESSAGE,
    Answer,
    DoneEvent,
    Question,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from app.exceptions.exception import GenerationError
from app.retrieval.protocol import RetrievedChunk, Retriever


def not_found(sources: list[RetrievedChunk]) -> Answer:
    return Answer(NOT_FOUND_MESSAGE, answered=False, citations=[], sources=sources)


def finish(text: str, sources: list[RetrievedChunk]) -> Answer:
    if not text.strip() or is_no_answer(text):
        return not_found(sources)
    return Answer(text.strip(), answered=True, citations=cite(text, sources), sources=sources)


async def answer_events(
    generator: AnswerGenerator, question: str, sources: list[RetrievedChunk]
) -> AsyncIterator[TokenEvent | DoneEvent]:
    if not sources:
        yield TokenEvent(NOT_FOUND_MESSAGE)
        yield DoneEvent(not_found(sources))
        return
    gate = SentinelGate(NO_ANSWER)
    parts: list[str] = []
    emitted = False
    async for delta in generator.astream(question, sources):
        parts.append(delta)
        released = gate.feed(delta)
        if released:
            emitted = True
            yield TokenEvent(released)
    tail = gate.finish()
    if tail.strip():
        emitted = True
        yield TokenEvent(tail)
    answer = finish("".join(parts), sources)
    if not emitted:
        yield TokenEvent(answer.text)
    yield DoneEvent(answer)


async def final_answer(events: AsyncIterator[StreamEvent]) -> Answer:
    async for event in events:
        if isinstance(event, DoneEvent):
            return event.answer
    raise GenerationError("answer stream ended without an answer")


class FastPathAnswerer:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    async def ask(self, question: Question) -> Answer:
        return await final_answer(self.stream(question))

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        sources = await asyncio.to_thread(self._retriever.retrieve, question.text)
        yield SourcesEvent(sources)
        async with aclosing(answer_events(self._generator, question.text, sources)) as events:
            async for event in events:
                yield event
