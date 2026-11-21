class AppError(Exception):
    status_code = 500
    code = "internal_error"
    public_message = "internal server error"

    @property
    def message(self) -> str:
        return str(self)

    @property
    def client_message(self) -> str:
        return self.message if self.status_code < 500 else self.public_message


class DatabaseUnavailable(AppError):
    status_code = 503
    code = "database_unavailable"
    public_message = "database unavailable"

    def __init__(self) -> None:
        super().__init__("database unavailable")


class NotReady(AppError):
    status_code = 503
    code = "not_ready"

    def __init__(self, failed: list[str]) -> None:
        super().__init__(f"not ready: {', '.join(failed)}")

    @property
    def client_message(self) -> str:
        return self.message


class InvalidQuery(AppError):
    status_code = 422
    code = "invalid_query"

    def __init__(self) -> None:
        super().__init__("query must not be empty")


class InvalidCursor(AppError):
    status_code = 400
    code = "invalid_cursor"

    def __init__(self) -> None:
        super().__init__("cursor is not valid")


class IdempotencyKeyReused(AppError):
    status_code = 422
    code = "idempotency_key_reused"

    def __init__(self) -> None:
        super().__init__("idempotency key was already used with a different request")


class IdempotencyKeyInProgress(AppError):
    status_code = 409
    code = "idempotency_request_in_progress"

    def __init__(self) -> None:
        super().__init__("a request with this idempotency key is still being processed")


class JobNotFound(AppError):
    status_code = 404
    code = "job_not_found"

    def __init__(self, job_id: int) -> None:
        super().__init__(f"job {job_id} does not exist")


class MigrationError(AppError):
    code = "migration_error"


class StorageError(AppError):
    status_code = 502
    code = "storage_error"
    public_message = "file storage unavailable"


class StoredFileMissing(StorageError):
    status_code = 404
    code = "stored_file_missing"

    def __init__(self) -> None:
        super().__init__("stored file does not exist")


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_error"
    public_message = "embedding service unavailable"


class EmbeddingDimensionMismatch(EmbeddingError, ValueError):
    status_code = 500
    code = "embedding_dimension_mismatch"
    public_message = "embedding configuration error"

    def __init__(self, expected: int, actual: int | None) -> None:
        super().__init__(
            f"embedding model produces {actual} dimensions, expected {expected}"
        )


class IngestionError(AppError):
    status_code = 422
    code = "ingestion_error"


class UnsupportedFileType(IngestionError):
    status_code = 415
    code = "unsupported_file_type"


class ParseError(IngestionError):
    code = "parse_error"


class InvalidUpload(IngestionError):
    status_code = 400
    code = "invalid_upload"


class InvalidSourceDocument(IngestionError):
    code = "invalid_source_document"


class FileTooLarge(IngestionError):
    status_code = 413
    code = "file_too_large"


class EmptyDocumentError(IngestionError):
    code = "empty_document"

    def __init__(self) -> None:
        super().__init__("no extractable text")


class EmbeddingFailed(IngestionError):
    status_code = 502
    code = "embedding_failed"
    public_message = "embedding service unavailable"


class DocumentNotFound(IngestionError):
    status_code = 404
    code = "document_not_found"

    def __init__(self, document_id: int) -> None:
        super().__init__(f"document {document_id} does not exist")


class DocumentNotRetryable(IngestionError):
    status_code = 409
    code = "document_not_retryable"

    def __init__(self, document_id: int, status: str) -> None:
        super().__init__(
            f"document {document_id} is {status}, only failed documents can be retried"
        )


class ContentMismatch(IngestionError):
    status_code = 409
    code = "content_mismatch"

    def __init__(self, document_id: int) -> None:
        super().__init__(f"content does not match stored document {document_id}")
