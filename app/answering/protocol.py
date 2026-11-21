from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.answering.citations import Citation
from app.retrieval.protocol import RetrievedChunk

DEFAULT_SCOPE = "default"
NOT_FOUND_MESSAGE ="I could not find the answer to that in the uploaded documents."
UNVERIFIED_MESSAGE = (
    "I found related passages but could not verify an answer against them, so I will not "
    "guess. Try rephrasing the question or check the cited documents directly."
)


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
    unanswered_parts: tuple[str, ...] = ()
    grounded: bool | None = None
    cached: bool = False
    thread_id: str | None = None


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
    thread_id: str | None = None
    scope: str = DEFAULT_SCOPE


class Answerer(Protocol):
    async def ask(self, question: Question) -> Answer: ...

    def stream(self, question: Question) -> AsyncIterator[StreamEvent]: ...
