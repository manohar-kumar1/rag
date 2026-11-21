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


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...
