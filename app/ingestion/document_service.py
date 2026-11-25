import logging
from dataclasses import dataclass

from app.exceptions.errors import StorageError
from app.ingestion.document_repository import DocumentRepository
from app.ingestion.records import DocumentRecord
from app.storage.protocol import FileStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentPage:
    items: list[DocumentRecord]
    next_before_id: int | None


class DocumentService:
    def __init__(self, repository: DocumentRepository, files: FileStore) -> None:
        self._repository = repository
        self._files = files

    def get(self, document_id: int) -> DocumentRecord:
        return self._repository.get(document_id)

    def list_page(self, limit: int, before_id: int | None) -> DocumentPage:
        rows = self._repository.list_page(limit + 1, before_id)
        items = rows[:limit]
        next_before_id = items[-1].id if len(rows) > limit else None
        return DocumentPage(items, next_before_id)

    def update_metadata(
        self, document_id: int, changes: dict[str, str | None]
    ) -> DocumentRecord:
        return self._repository.update_metadata(document_id, changes)

    def retry(self, document_id: int) -> DocumentRecord:
        return self._repository.retry(document_id)

    def delete(self, document_id: int) -> None:
        storage_key = self._repository.delete(document_id)
        if storage_key is None:
            return
        try:
            self._files.delete(storage_key)
        except StorageError:
            logger.exception(
                "stored original for deleted document %s was not removed", document_id
            )
