from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.answering.citations import Citation
from app.retrieval.protocol import RetrievedChunk

NOT_FOUND_MESSAGE = "I could not find the answer to that in the uploaded documents."


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Answer:
    text: str
    answered: bool
    citations: list[Citation]
    sources: list[RetrievedChunk]
    route: str | None = None
    search_query: str | None = None


@dataclass(frozen=True)
class SourcesEvent:
    sources: list[RetrievedChunk]


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    answer: Answer


StreamEvent = SourcesEvent | TokenEvent | DoneEvent


@dataclass(frozen=True)
class Question:
    text: str
    history: Sequence[Turn] = field(default_factory=tuple)


class Answerer(Protocol):
    async def ask(self, question: Question) -> Answer: ...

    def stream(self, question: Question) -> AsyncIterator[StreamEvent]: ...
