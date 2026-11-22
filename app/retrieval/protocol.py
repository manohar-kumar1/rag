from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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


class Retriever(Protocol):
    def retrieve(
        self, query: str, document_ids: Sequence[int] = ()
    ) -> list[RetrievedChunk]: ...
