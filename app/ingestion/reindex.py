import sys
from collections.abc import Callable, Iterator, Sequence

from app.composition import build_container
from app.core.config import get_settings
from app.exceptions.errors import AppError
from app.exceptions.handlers import report_cli_error
from app.ingestion.document_repository import DocumentRepository
from app.ingestion.pipeline import IngestionResult
from app.ingestion.records import DocumentRecord
from app.storage.protocol import FileStore

PAGE_SIZE = 100
READY = "ready"


def stored_documents(repository: DocumentRepository) -> Iterator[DocumentRecord]:
    before_id: int | None = None
    while True:
        page = repository.list_page(PAGE_SIZE, before_id)
        yield from page
        if len(page) < PAGE_SIZE:
            return
        before_id = page[-1].id


def reindex(
    repository: DocumentRepository,
    files: FileStore,
    reingest: Callable[[int, bytes, str | None], IngestionResult],
    report: Callable[[str], None],
) -> int:
    failures = 0
    for record in stored_documents(repository):
        if record.status != READY or record.storage_key is None:
            report(f"skipped {record.id} {record.source} ({record.status})")
            continue
        try:
            result = reingest(record.id, files.get(record.storage_key), record.title)
        except AppError as error:
            failures += 1
            report(f"failed {record.id} {record.source}: {error.client_message}")
        else:
            report(f"reindexed {record.id} {record.source}: {result.chunk_count} chunks")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    container = build_container(settings)
    container.pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    try:
        failures = reindex(
            container.documents,
            container.files,
            lambda document_id, data, title: container.ingestion.reingest(
                document_id, data, title=title
            ),
            print,
        )
    except AppError as error:
        return report_cli_error(error)
    finally:
        container.pool.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
