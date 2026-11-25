from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.answering.citations import Citation
from app.answering.protocol import Answer, Question, Turn
from app.retrieval.filters import RetrievalFilter
from app.retrieval.protocol import RetrievedChunk

MAX_QUESTION_CHARS = 2000
MAX_TURN_CHARS = 8000
MAX_HISTORY_TURNS = 20
MAX_SCOPED_DOCUMENTS = 50
MAX_FILTER_CHARS = 100


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TURN_CHARS)


class QueryFilters(BaseModel):
    department: str | None = Field(default=None, min_length=1, max_length=MAX_FILTER_CHARS)
    product: str | None = Field(default=None, min_length=1, max_length=MAX_FILTER_CHARS)
    version: str | None = Field(default=None, min_length=1, max_length=MAX_FILTER_CHARS)
    doc_type: str | None = Field(default=None, min_length=1, max_length=MAX_FILTER_CHARS)
    updated_after: date | None = None
    updated_before: date | None = None


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[TurnIn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    thread_id: UUID | None = Field(
        default=None,
        description="Server-stored conversation. When set, history comes from the server "
        "and `history` must be empty.",
    )

    document_ids: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=list,
        max_length=MAX_SCOPED_DOCUMENTS,
        description="Answer only from these documents. Empty means every document.",
    )
    filters: "QueryFilters" = Field(
        default_factory=lambda: QueryFilters(),
        description="Answer only from documents whose metadata matches every set field.",
    )

    @model_validator(mode="after")
    def history_or_thread(self) -> "QueryIn":
        if self.thread_id is not None and self.history:
            raise ValueError("send either thread_id or history, not both")
        return self

    def to_question(self) -> Question:
        return Question(
            text=self.question,
            history=tuple(Turn(turn.role, turn.content) for turn in self.history),
            thread_id=None if self.thread_id is None else str(self.thread_id),
            filters=RetrievalFilter(
                document_ids=tuple(self.document_ids),
                department=self.filters.department,
                product=self.filters.product,
                version=self.filters.version,
                doc_type=self.filters.doc_type,
                updated_after=self.filters.updated_after,
                updated_before=self.filters.updated_before,
            ),
        )


class SourceOut(BaseModel):
    marker: int
    chunk_id: int
    document_id: int
    source: str
    title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    heading_path: str
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
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            heading_path=chunk.heading_path,
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
    unanswered_parts: list[str]
    grounded: bool | None
    cached: bool
    thread_id: str | None

    @classmethod
    def from_answer(cls, answer: Answer) -> "QueryOut":
        return cls(
            answer=answer.text,
            answered=answer.answered,
            citations=[SourceOut.from_citation(c) for c in answer.citations],
            route=answer.route,
            search_query=answer.search_query,
            unanswered_parts=list(answer.unanswered_parts),
            grounded=answer.grounded,
            cached=answer.cached,
            thread_id=answer.thread_id,
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


class ProgressOut(BaseModel):
    step: str
    status: Literal["running", "done"]
