from typing import Literal

from pydantic import BaseModel, Field

from app.answering.citations import Citation
from app.answering.protocol import Answer, Question, Turn
from app.retrieval.protocol import RetrievedChunk

MAX_QUESTION_CHARS = 2000
MAX_TURN_CHARS = 8000
MAX_HISTORY_TURNS = 20


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TURN_CHARS)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[TurnIn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    def to_question(self) -> Question:
        return Question(
            text=self.question,
            history=tuple(Turn(turn.role, turn.content) for turn in self.history),
        )


class SourceOut(BaseModel):
    marker: int
    chunk_id: int
    document_id: int
    source: str
    title: str | None
    chunk_index: int
    text: str

    @classmethod
    def from_chunk(cls, marker: int, chunk: RetrievedChunk) -> "SourceOut":
        return cls(
            marker=marker,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source=chunk.source,
            title=chunk.title,
            chunk_index=chunk.chunk_index,
            text=chunk.content,
        )

    @classmethod
    def from_citation(cls, citation: Citation) -> "SourceOut":
        return cls.from_chunk(citation.marker, citation.chunk)


class QueryOut(BaseModel):
    answer: str
    answered: bool
    citations: list[SourceOut]
    route: str | None
    search_query: str | None

    @classmethod
    def from_answer(cls, answer: Answer) -> "QueryOut":
        return cls(
            answer=answer.text,
            answered=answer.answered,
            citations=[SourceOut.from_citation(c) for c in answer.citations],
            route=answer.route,
            search_query=answer.search_query,
        )


class SourcesOut(BaseModel):
    sources: list[SourceOut]

    @classmethod
    def from_chunks(cls, chunks: list[RetrievedChunk]) -> "SourcesOut":
        return cls(
            sources=[SourceOut.from_chunk(n, c) for n, c in enumerate(chunks, start=1)]
        )


class TokenOut(BaseModel):
    text: str
