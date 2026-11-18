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


class MigrationError(AppError):
    code = "migration_error"


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


class ContentMismatch(IngestionError):
    status_code = 409
    code = "content_mismatch"

    def __init__(self, document_id: int) -> None:
        super().__init__(f"content does not match stored document {document_id}")
