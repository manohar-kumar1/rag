from typing import Protocol

import psycopg
from psycopg.rows import class_row

from app.exceptions.exception import DocumentNotFound, DocumentNotRetryable
from app.ingestion.records import DocumentRecord, NewUpload
from app.ingestion.status import IngestionStage
from app.ingestion.store import ConnectionSource

COLUMNS = (
    "id, source, title, content_hash, status, error, original_filename, content_type, "
    "size_bytes, storage_key, created_at, processing_started_at, processed_at, processing_stage"
)
SELECT_DOCUMENT = f"select {COLUMNS} from documents"
FIND_BY_HASH = f"{SELECT_DOCUMENT} where content_hash = %s"
GET_BY_ID = f"{SELECT_DOCUMENT} where id = %s"
INSERT_PENDING = (
    "insert into documents (source, title, content_hash, metadata, status, "
    "original_filename, content_type, size_bytes, storage_key) "
    "values (%s, %s, %s, '{}'::jsonb, 'pending', %s, %s, %s, %s) returning id"
)
CLAIM_NEXT = (
    "update documents set status = 'processing', processing_started_at = now(), "
    "processing_stage = 'parsing', error = null where id = ("
    "select id from documents where status = 'pending' "
    "order by created_at, id for update skip locked limit 1) "
    f"returning {COLUMNS}"
)
RECLAIM_STALE = (
    "update documents set status = 'pending', processing_started_at = null, "
    "processing_stage = null where status = 'processing' "
    "and processing_started_at < now() - make_interval(secs => %s)"
)
MARK_READY = (
    "update documents set status = 'ready', error = null, processing_stage = null, "
    "processed_at = now() "
    "where id = %s and status = 'processing'"
)
MARK_FAILED = (
    "update documents set status = 'failed', error = %s, processing_stage = null, "
    "processed_at = now() "
    "where id = %s and status = 'processing'"
)
RETRY = (
    "update documents set status = 'pending', error = null, "
    "processing_started_at = null, processed_at = null, processing_stage = null "
    f"where id = %s and status = 'failed' returning {COLUMNS}"
)
SET_STAGE = (
    "update documents set processing_stage = %s where id = %s and status = 'processing'"
)


LIST_PAGE = (
    f"{SELECT_DOCUMENT} where (%s::bigint is null or id < %s) order by id desc limit %s"
)
DELETE_DOCUMENT = "delete from documents where id = %s returning storage_key"


class DocumentRepository(Protocol):
    def find_by_hash(self, content_hash: str) -> DocumentRecord | None: ...

    def create_pending(self, upload: NewUpload) -> tuple[DocumentRecord, bool]: ...

    def get(self, document_id: int) -> DocumentRecord: ...

    def retry(self, document_id: int) -> DocumentRecord: ...

    def list_page(self, limit: int, before_id: int | None) -> list[DocumentRecord]: ...

    def delete(self, document_id: int) -> str | None: ...


class WorkQueue(Protocol):
    def claim_next(self) -> DocumentRecord | None: ...

    def reclaim_stale(self, stale_after_seconds: float) -> int: ...

    def set_stage(self, document_id: int, stage: IngestionStage) -> None: ...

    def mark_ready(self, document_id: int) -> None: ...

    def mark_failed(self, document_id: int, reason: str) -> None: ...


def fetch_one(
    conn: psycopg.Connection, query: str, params: tuple
) -> DocumentRecord | None:
    with conn.cursor(row_factory=class_row(DocumentRecord)) as cursor:
        return cursor.execute(query, params).fetchone()


class PostgresDocumentRepository:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def find_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._source.connection() as conn:
            return fetch_one(conn, FIND_BY_HASH, (content_hash,))

    def create_pending(self, upload: NewUpload) -> tuple[DocumentRecord, bool]:
        with self._source.connection() as conn:
            try:
                with conn.transaction():
                    document_id = conn.execute(
                        INSERT_PENDING,
                        (
                            upload.source,
                            upload.title,
                            upload.content_hash,
                            upload.original_filename,
                            upload.content_type,
                            upload.size_bytes,
                            upload.storage_key,
                        ),
                    ).fetchone()[0]
            except psycopg.errors.UniqueViolation:
                existing = fetch_one(conn, FIND_BY_HASH, (upload.content_hash,))
                if existing is None:
                    raise
                return existing, False
            return fetch_one(conn, GET_BY_ID, (document_id,)), True

    def get(self, document_id: int) -> DocumentRecord:
        with self._source.connection() as conn:
            record = fetch_one(conn, GET_BY_ID, (document_id,))
        if record is None:
            raise DocumentNotFound(document_id)
        return record

    def retry(self, document_id: int) -> DocumentRecord:
        with self._source.connection() as conn:
            record = fetch_one(conn, RETRY, (document_id,))
            if record is not None:
                return record
            current = fetch_one(conn, GET_BY_ID, (document_id,))
        if current is None:
            raise DocumentNotFound(document_id)
        raise DocumentNotRetryable(document_id, current.status)

    def list_page(self, limit: int, before_id: int | None) -> list[DocumentRecord]:
        with (
            self._source.connection() as conn,
            conn.cursor(row_factory=class_row(DocumentRecord)) as cursor,
        ):
            return cursor.execute(LIST_PAGE, (before_id, before_id, limit)).fetchall()

    def delete(self, document_id: int) -> str | None:
        with self._source.connection() as conn:
            row = conn.execute(DELETE_DOCUMENT, (document_id,)).fetchone()
        if row is None:
            raise DocumentNotFound(document_id)
        return row[0]

    def claim_next(self) -> DocumentRecord | None:
        with self._source.connection() as conn:
            return fetch_one(conn, CLAIM_NEXT, ())

    def reclaim_stale(self, stale_after_seconds: float) -> int:
        with self._source.connection() as conn:
            return conn.execute(RECLAIM_STALE, (stale_after_seconds,)).rowcount

    def set_stage(self, document_id: int, stage: IngestionStage) -> None:
        with self._source.connection() as conn:
            conn.execute(SET_STAGE, (stage.value, document_id))

    def mark_ready(self, document_id: int) -> None:
        with self._source.connection() as conn:
            conn.execute(MARK_READY, (document_id,))

    def mark_failed(self, document_id: int, reason: str) -> None:
        with self._source.connection() as conn:
            conn.execute(MARK_FAILED, (reason, document_id))
