from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import psycopg
from psycopg.types.json import Jsonb

from app.db.vector import to_vector_literal
from app.exceptions.exception import ContentMismatch, DocumentNotFound

FIND_BY_HASH = (
    "select d.id, (select count(*) from chunks c where c.document_id = d.id) "
    "from documents d where d.content_hash = %s"
)
INSERT_DOCUMENT = (
    "insert into documents (source, title, content_hash, metadata) "
    "values (%s, %s, %s, %s) returning id"
)
LOCK_DOCUMENT = "select content_hash from documents where id = %s for update"
DELETE_CHUNKS = "delete from chunks where document_id = %s"
INSERT_CHUNK = (
    "insert into chunks (document_id, chunk_index, content, embedding) "
    "values (%s, %s, %s, %s::extensions.vector)"
)


@dataclass(frozen=True)
class NewDocument:
    source: str
    title: str | None
    content_hash: str
    metadata: dict


@dataclass(frozen=True)
class StoredDocument:
    id: int
    chunk_count: int
    created: bool


class ConnectionSource(Protocol):
    def connection(self) -> AbstractContextManager[psycopg.Connection]: ...


class DocumentStore(Protocol):
    def find_by_hash(self, content_hash: str) -> StoredDocument | None: ...

    def add_document(
        self,
        document: NewDocument,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> StoredDocument: ...

    def replace_chunks(
        self,
        document_id: int,
        content_hash: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> None:
    rows = [
        (document_id, index, content, to_vector_literal(vector))
        for index, (content, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_CHUNK, rows)


def find_existing(conn: psycopg.Connection, content_hash: str) -> StoredDocument | None:
    row = conn.execute(FIND_BY_HASH, (content_hash,)).fetchone()
    return None if row is None else StoredDocument(row[0], row[1], created=False)


class PostgresDocumentStore:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def find_by_hash(self, content_hash: str) -> StoredDocument | None:
        with self._source.connection() as conn:
            return find_existing(conn, content_hash)

    def add_document(
        self,
        document: NewDocument,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> StoredDocument:
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
                        ),
                    ).fetchone()[0]
                    insert_chunks(conn, document_id, chunks, vectors)
            except psycopg.errors.UniqueViolation:
                existing = find_existing(conn, document.content_hash)
                if existing is None:
                    raise
                return existing
            return StoredDocument(document_id, len(chunks), created=True)

    def replace_chunks(
        self,
        document_id: int,
        content_hash: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        with self._source.connection() as conn, conn.transaction():
            row = conn.execute(LOCK_DOCUMENT, (document_id,)).fetchone()
            if row is None:
                raise DocumentNotFound(document_id)
            if row[0] != content_hash:
                raise ContentMismatch(document_id)
            conn.execute(DELETE_CHUNKS, (document_id,))
            insert_chunks(conn, document_id, chunks, vectors)
