from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePath

from app.exceptions.errors import (
    DocumentNotRetryable,
    FileTooLarge,
    InvalidUpload,
    UnsupportedFileType,
)
from app.ingestion.document_repository import DocumentRepository
from app.ingestion.file_types import MIME_TYPES, FileType, detect_file_type
from app.ingestion.records import DocumentRecord, NewUpload
from app.ingestion.status import DocumentStatus
from app.storage.keys import new_storage_key
from app.storage.protocol import FileStore

EXTENSION_TYPES = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TEXT,
    ".md": FileType.TEXT,
}
MAX_FILENAME_LENGTH = 255


@dataclass(frozen=True)
class UploadResult:
    record: DocumentRecord
    created: bool
    replayed: bool = False


def clean_filename(filename: str) -> str:
    name = PurePath(filename.replace("\\", "/")).name.strip()
    if not name:
        raise InvalidUpload("a file name is required")
    return name[:MAX_FILENAME_LENGTH]


def expected_type(filename: str) -> FileType:
    file_type = EXTENSION_TYPES.get(PurePath(filename).suffix.lower())
    if file_type is None:
        supported = ", ".join(sorted(EXTENSION_TYPES))
        raise UnsupportedFileType(
            f"unsupported file extension, use one of: {supported}"
        )
    return file_type


class UploadService:
    def __init__(
        self, repository: DocumentRepository, files: FileStore, max_upload_bytes: int
    ) -> None:
        self._repository = repository
        self._files = files
        self._max_upload_bytes = max_upload_bytes

    def submit(self, data: bytes, filename: str) -> UploadResult:
        name = clean_filename(filename)
        if len(data) > self._max_upload_bytes:
            raise FileTooLarge("file exceeds the upload limit")
        file_type = detect_file_type(data)
        if expected_type(name) is not file_type:
            raise UnsupportedFileType("file content does not match its extension")
        content_hash = sha256(data).hexdigest()
        existing = self._repository.find_by_hash(content_hash)
        if existing is not None:
            return UploadResult(self._requeue_if_failed(existing), False)
        return self._store(data, name, file_type, content_hash)

    def _requeue_if_failed(self, record: DocumentRecord) -> DocumentRecord:
        if record.status != DocumentStatus.FAILED:
            return record
        try:
            return self._repository.retry(record.id)
        except DocumentNotRetryable:
            return self._repository.get(record.id)

    def _store(
        self, data: bytes, name: str, file_type: FileType, content_hash: str
    ) -> UploadResult:
        key = new_storage_key()
        self._files.put(key, data, MIME_TYPES[file_type])
        try:
            record, created = self._repository.create_pending(
                NewUpload(
                    source=name,
                    title=PurePath(name).stem,
                    content_hash=content_hash,
                    original_filename=name,
                    content_type=MIME_TYPES[file_type],
                    size_bytes=len(data),
                    storage_key=key,
                    doc_type=file_type.value,
                )
            )
        except Exception:
            self._files.delete(key)
            raise
        if not created:
            self._files.delete(key)
        return UploadResult(record, created)
