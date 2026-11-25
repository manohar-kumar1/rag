from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from app.db.connection_source import ConnectionSource
from app.exceptions.errors import IdempotencyKeyInProgress, IdempotencyKeyReused
from app.ingestion.document_repository import DocumentRepository
from app.ingestion.upload_service import UploadResult

DELETE_EXPIRED = "delete from idempotency_keys where key = %s and expires_at < now()"
INSERT_KEY = (
    "insert into idempotency_keys (key, request_hash, state, expires_at) "
    "values (%s, %s, 'in_progress', now() + make_interval(secs => %s)) "
    "on conflict (key) do nothing returning key"
)
SELECT_KEY = "select request_hash, state, document_id, created from idempotency_keys where key = %s"
COMPLETE_KEY = (
    "update idempotency_keys set state = 'completed', document_id = %s, created = %s, "
    "expires_at = now() + make_interval(secs => %s) where key = %s"
)
RELEASE_KEY = "delete from idempotency_keys where key = %s and state = 'in_progress'"


class ClaimOutcome(StrEnum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    MISMATCH = "mismatch"
    REPLAY = "replay"


@dataclass(frozen=True)
class IdempotencyClaim:
    outcome: ClaimOutcome
    document_id: int | None = None
    created: bool | None = None


class IdempotencyStore(Protocol):
    def begin(self, key: str, request_hash: str) -> IdempotencyClaim: ...

    def complete(self, key: str, document_id: int, created: bool) -> None: ...

    def release(self, key: str) -> None: ...


class UploadSubmitter(Protocol):
    def submit(self, data: bytes, filename: str) -> UploadResult: ...


class PostgresIdempotencyStore:
    def __init__(
        self, source: ConnectionSource, ttl_seconds: float, lock_seconds: float
    ) -> None:
        self._source = source
        self._ttl_seconds = ttl_seconds
        self._lock_seconds = lock_seconds

    def begin(self, key: str, request_hash: str) -> IdempotencyClaim:
        with self._source.connection() as conn, conn.transaction():
            conn.execute(DELETE_EXPIRED, (key,))
            inserted = conn.execute(
                INSERT_KEY, (key, request_hash, self._lock_seconds)
            ).fetchone()
            if inserted is not None:
                return IdempotencyClaim(ClaimOutcome.NEW)
            stored_hash, state, document_id, created = conn.execute(
                SELECT_KEY, (key,)
            ).fetchone()
        if stored_hash != request_hash:
            return IdempotencyClaim(ClaimOutcome.MISMATCH)
        if state == "in_progress":
            return IdempotencyClaim(ClaimOutcome.IN_PROGRESS)
        return IdempotencyClaim(ClaimOutcome.REPLAY, document_id, created)

    def complete(self, key: str, document_id: int, created: bool) -> None:
        with self._source.connection() as conn:
            conn.execute(COMPLETE_KEY, (document_id, created, self._ttl_seconds, key))

    def release(self, key: str) -> None:
        with self._source.connection() as conn:
            conn.execute(RELEASE_KEY, (key,))


def request_fingerprint(data: bytes, filename: str) -> str:
    return sha256(filename.encode() + b"\0" + data).hexdigest()


class IdempotentUploads:
    def __init__(
        self,
        uploads: UploadSubmitter,
        keys: IdempotencyStore,
        repository: DocumentRepository,
    ) -> None:
        self._uploads = uploads
        self._keys = keys
        self._repository = repository

    def submit(self, data: bytes, filename: str, key: str | None) -> UploadResult:
        if key is None:
            return self._uploads.submit(data, filename)
        claim = self._keys.begin(key, request_fingerprint(data, filename))
        match claim.outcome:
            case ClaimOutcome.MISMATCH:
                raise IdempotencyKeyReused
            case ClaimOutcome.IN_PROGRESS:
                raise IdempotencyKeyInProgress
            case ClaimOutcome.REPLAY:
                record = self._repository.get(claim.document_id)
                return UploadResult(record, bool(claim.created), replayed=True)
        return self._submit_and_record(data, filename, key)

    def _submit_and_record(self, data: bytes, filename: str, key: str) -> UploadResult:
        try:
            result = self._uploads.submit(data, filename)
        except Exception:
            self._keys.release(key)
            raise
        self._keys.complete(key, result.record.id, result.created)
        return result
