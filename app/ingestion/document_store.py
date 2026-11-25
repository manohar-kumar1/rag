from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import psycopg
from psycopg.types.json import Jsonb

from app.db.connection_source import ConnectionSource
from app.db.vector import from_vector_literal, to_vector_literal
from app.exceptions.errors import ContentMismatch, DocumentNotFound
from app.ingestion.external_sources import SyncOutcome

FIND_BY_HASH = (
    "select d.id, (select count(*) from chunks c where c.document_id = d.id) "
    "from documents d where d.content_hash = %s"
)
INSERT_DOCUMENT = (
    "insert into documents (source, title, content_hash, metadata, status, "
    "original_filename, content_type, size_bytes, doc_type, processed_at) "
    "values (%s, %s, %s, %s, 'ready', %s, %s, %s, %s, now()) returning id"
)
FIND_BY_EXTERNAL_REF = (
    "select d.id, d.content_hash, "
    "(select count(*) from chunks c where c.document_id = d.id) "
    "from documents d where d.external_ref = %s"
)
LOCK_BY_EXTERNAL_REF = (
    "select id, content_hash from documents where external_ref = %s for update"
)
INSERT_EXTERNAL = (
    "insert into documents (source, title, content_hash, metadata, status, content_type, "
    "size_bytes, external_ref, doc_type, processed_at) "
    "values (%s, %s, %s, %s, 'ready', 'text/plain', %s, %s, 'text', now()) returning id"
)
UPDATE_EXTERNAL = (
    "update documents set title = %s, content_hash = %s, metadata = %s, size_bytes = %s, "
    "status = 'ready', error = null, processed_at = now() where id = %s"
)
LOCK_DOCUMENT = "select content_hash from documents where id = %s for update"
DELETE_SECTIONS = "delete from sections where document_id = %s"
INSERT_SECTIONS = (
    "insert into sections (document_id, section_index, heading_path, content, token_count, "
    "page_start, page_end) select %s, * from unnest(%s::int[], %s::text[], %s::text[], "
    "%s::int[], %s::int[], %s::int[]) returning section_index, id"
)
INSERT_CHUNK = (
    "insert into chunks (document_id, section_id, chunk_index, content, context, heading_path, "
    "token_count, chunk_hash, page_start, page_end, embedding) "
    "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::extensions.vector)"
)
EXISTING_VECTORS = (
    "select chunk_hash, embedding::text from chunks "
    "where document_id = %s and chunk_hash is not null"
)


@dataclass(frozen=True)
class ChunkText:
    content: str
    context: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: str = ""
    token_count: int | None = None
    chunk_hash: str | None = None
    section_index: int | None = None

    @property
    def embedding_text(self) -> str:
        return self.content if not self.context else f"{self.context}\n\n{self.content}"


@dataclass(frozen=True)
class SectionText:
    index: int
    heading_path: str
    content: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class StoredContent:
    chunks: Sequence[ChunkText]
    vectors: Sequence[Sequence[float]]
    sections: Sequence[SectionText] = field(default_factory=tuple)


@dataclass(frozen=True)
class NewDocument:
    source: str
    title: str | None
    content_hash: str
    metadata: dict
    original_filename: str | None
    content_type: str | None
    size_bytes: int | None
    doc_type: str | None = None


@dataclass(frozen=True)
class StoredDocument:
    id: int
    chunk_count: int
    created: bool


@dataclass(frozen=True)
class NewExternalDocument:
    reference: str
    title: str | None
    content_hash: str
    metadata: dict
    size_bytes: int


@dataclass(frozen=True)
class ExternalState:
    id: int
    content_hash: str
    chunk_count: int


@dataclass(frozen=True)
class StoredExternal:
    id: int
    chunk_count: int
    outcome: SyncOutcome


class DocumentStore(Protocol):
    def find_by_hash(self, content_hash: str) -> StoredDocument | None: ...

    def add_document(self, document: NewDocument, content: StoredContent) -> StoredDocument: ...

    def replace_chunks(
        self, document_id: int, content_hash: str, content: StoredContent
    ) -> None: ...

    def existing_vectors(self, document_id: int) -> dict[str, list[float]]: ...

    def find_by_external_ref(self, reference: str) -> ExternalState | None: ...

    def upsert_external(
        self, document: NewExternalDocument, content: StoredContent
    ) -> StoredExternal: ...


def insert_sections(
    conn: psycopg.Connection, document_id: int, sections: Sequence[SectionText]
) -> dict[int, int]:
    if not sections:
        return {}
    rows = conn.execute(
        INSERT_SECTIONS,
        (
            document_id,
            [section.index for section in sections],
            [section.heading_path for section in sections],
            [section.content for section in sections],
            [section.token_count for section in sections],
            [section.page_start for section in sections],
            [section.page_end for section in sections],
        ),
    ).fetchall()
    return dict(rows)


def insert_content(conn: psycopg.Connection, document_id: int, content: StoredContent) -> None:
    section_ids = insert_sections(conn, document_id, content.sections)
    rows = [
        (
            document_id,
            section_ids.get(chunk.section_index) if chunk.section_index is not None else None,
            index,
            chunk.content,
            chunk.context,
            chunk.heading_path,
            chunk.token_count,
            chunk.chunk_hash,
            chunk.page_start,
            chunk.page_end,
            to_vector_literal(vector),
        )
        for index, (chunk, vector) in enumerate(zip(content.chunks, content.vectors, strict=True))
    ]
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_CHUNK, rows)


def clear_content(conn: psycopg.Connection, document_id: int) -> None:
    conn.execute("delete from chunks where document_id = %s", (document_id,))
    conn.execute(DELETE_SECTIONS, (document_id,))


def find_existing(conn: psycopg.Connection, content_hash: str) -> StoredDocument | None:
    row = conn.execute(FIND_BY_HASH, (content_hash,)).fetchone()
    return None if row is None else StoredDocument(row[0], row[1], created=False)


class PostgresDocumentStore:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def find_by_hash(self, content_hash: str) -> StoredDocument | None:
        with self._source.connection() as conn:
            return find_existing(conn, content_hash)

    def existing_vectors(self, document_id: int) -> dict[str, list[float]]:
        with self._source.connection() as conn:
            rows = conn.execute(EXISTING_VECTORS, (document_id,)).fetchall()
        return {chunk_hash: from_vector_literal(vector) for chunk_hash, vector in rows}

    def add_document(self, document: NewDocument, content: StoredContent) -> StoredDocument:
        with self._source.connection() as conn:
            try:
                with conn.transaction():
                    document_id = conn.execute(
                        INSERT_DOCUMENT,
                        (
                            document.source,
                            document.title,
                            document.content_hash,
                            Jsonb(document.metadata),
                            document.original_filename,
                            document.content_type,
                            document.size_bytes,
                            document.doc_type,
                        ),
                    ).fetchone()[0]
                    insert_content(conn, document_id, content)
            except psycopg.errors.UniqueViolation:
                existing = find_existing(conn, document.content_hash)
                if existing is None:
                    raise
                return existing
            return StoredDocument(document_id, len(content.chunks), created=True)

    def replace_chunks(
        self, document_id: int, content_hash: str, content: StoredContent
    ) -> None:
        with self._source.connection() as conn, conn.transaction():
            row = conn.execute(LOCK_DOCUMENT, (document_id,)).fetchone()
            if row is None:
                raise DocumentNotFound(document_id)
            if row[0] != content_hash:
                raise ContentMismatch(document_id)
            clear_content(conn, document_id)
            insert_content(conn, document_id, content)

    def find_by_external_ref(self, reference: str) -> ExternalState | None:
        with self._source.connection() as conn:
            row = conn.execute(FIND_BY_EXTERNAL_REF, (reference,)).fetchone()
        return None if row is None else ExternalState(*row)

    def upsert_external(
        self, document: NewExternalDocument, content: StoredContent
    ) -> StoredExternal:
        with self._source.connection() as conn:
            try:
                with conn.transaction():
                    row = conn.execute(
                        LOCK_BY_EXTERNAL_REF, (document.reference,)
                    ).fetchone()
                    if row is None:
                        document_id = self._insert_external(conn, document)
                        outcome: SyncOutcome = "created"
                    else:
                        document_id = row[0]
                        self._update_external(conn, document_id, document)
                        outcome = "updated"
                    insert_content(conn, document_id, content)
            except psycopg.errors.UniqueViolation:
                existing = find_existing(conn, document.content_hash)
                if existing is None:
                    raise
                return StoredExternal(existing.id, existing.chunk_count, "unchanged")
            return StoredExternal(document_id, len(content.chunks), outcome)

    @staticmethod
    def _insert_external(
        conn: psycopg.Connection, document: NewExternalDocument
    ) -> int:
        return conn.execute(
            INSERT_EXTERNAL,
            (
                document.reference,
                document.title,
                document.content_hash,
                Jsonb(document.metadata),
                document.size_bytes,
                document.reference,
            ),
        ).fetchone()[0]

    @staticmethod
    def _update_external(
        conn: psycopg.Connection, document_id: int, document: NewExternalDocument
    ) -> None:
        conn.execute(
            UPDATE_EXTERNAL,
            (
                document.title,
                document.content_hash,
                Jsonb(document.metadata),
                document.size_bytes,
                document_id,
            ),
        )
        clear_content(conn, document_id)
