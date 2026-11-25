from dataclasses import dataclass
from typing import Protocol

from app.retrieval.filters import EMPTY_FILTER, RetrievalFilter


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    source: str
    title: str | None
    score: float
    vector_rank: int | None
    fts_rank: int | None
    rerank_score: float | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: str = ""
    section_id: int | None = None
    passage: str | None = None

    @property
    def prompt_text(self) -> str:
        return self.passage or self.content


class Retriever(Protocol):
    def retrieve(
        self, query: str, filters: RetrievalFilter = EMPTY_FILTER
    ) -> list[RetrievedChunk]: ...
