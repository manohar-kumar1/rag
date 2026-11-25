from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from app.db.connection_source import ConnectionSource
from app.retrieval.filters import EMPTY_FILTER, RetrievalFilter
from app.retrieval.protocol import RetrievedChunk, Retriever
from app.retrieval.stage_log import log_stage

READ_SECTIONS = (
    "select id, heading_path, content, token_count, page_start, page_end "
    "from sections where id = any(%s)"
)


@dataclass(frozen=True)
class StoredSection:
    id: int
    heading_path: str
    content: str
    token_count: int
    page_start: int | None
    page_end: int | None


class SectionReader(Protocol):
    def read(self, section_ids: Sequence[int]) -> dict[int, StoredSection]: ...


class PostgresSectionReader:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def read(self, section_ids: Sequence[int]) -> dict[int, StoredSection]:
        if not section_ids:
            return {}
        with self._source.connection() as conn:
            rows = conn.execute(READ_SECTIONS, (list(section_ids),)).fetchall()
        return {row[0]: StoredSection(*row) for row in rows}


class AssemblingRetriever:
    def __init__(self, inner: Retriever, sections: SectionReader, parent_max_tokens: int) -> None:
        self._inner = inner
        self._sections = sections
        self._parent_max_tokens = parent_max_tokens

    def retrieve(
        self, query: str, filters: RetrievalFilter = EMPTY_FILTER
    ) -> list[RetrievedChunk]:
        chunks = self._inner.retrieve(query, filters)
        if self._parent_max_tokens <= 0 or not chunks:
            return chunks
        found = self._sections.read([c.section_id for c in chunks if c.section_id is not None])
        assembled: list[RetrievedChunk] = []
        taken: set[int] = set()
        merged: list[int] = []
        for chunk in chunks:
            section = found.get(chunk.section_id) if chunk.section_id is not None else None
            if section is None or section.token_count > self._parent_max_tokens:
                assembled.append(chunk)
            elif section.id in taken:
                merged.append(chunk.chunk_id)
            else:
                taken.add(section.id)
                assembled.append(replace(chunk, passage=section.content))
        log_stage(
            "assemble",
            kept=[chunk.chunk_id for chunk in assembled],
            expanded_sections=sorted(taken),
            merged_chunks=merged,
        )
        return assembled
