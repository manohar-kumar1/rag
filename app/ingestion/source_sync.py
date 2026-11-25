import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.exceptions.errors import AppError
from app.ingestion.external_sources import Source, SourceDocument, SyncResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncFailure:
    reference: str
    reason: str


@dataclass(frozen=True)
class SyncReport:
    results: list[SyncResult] = field(default_factory=list)
    failures: list[SyncFailure] = field(default_factory=list)


class SourceIngester(Protocol):
    def ingest_source_document(self, document: SourceDocument) -> SyncResult: ...


class SourceSyncer:
    def __init__(self, pipeline: SourceIngester) -> None:
        self._pipeline = pipeline

    def sync(self, source: Source) -> SyncReport:
        results: list[SyncResult] = []
        failures: list[SyncFailure] = []
        for document in source.documents():
            try:
                results.append(self._pipeline.ingest_source_document(document))
            except AppError as error:
                logger.warning(
                    "source document %s failed: %s", document.reference, error.code
                )
                failures.append(SyncFailure(document.reference, error.client_message))
        return SyncReport(results, failures)
