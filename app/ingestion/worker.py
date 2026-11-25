import logging
import threading
from typing import Protocol

from app.exceptions.errors import AppError
from app.ingestion.document_repository import WorkQueue
from app.ingestion.pipeline import StageListener
from app.ingestion.records import DocumentRecord
from app.storage.protocol import FileStore

logger = logging.getLogger(__name__)
UNEXPECTED_FAILURE = "internal server error"
MISSING_FILE_KEY = "stored file reference is missing"


class Reingester(Protocol):
    def reingest(
        self, document_id: int, data: bytes, on_stage: StageListener, title: str | None
    ): ...


class IngestionWorker:
    def __init__(
        self,
        queue: WorkQueue,
        files: FileStore,
        pipeline: Reingester,
        poll_interval_seconds: float,
        stale_after_seconds: float,
        concurrency: int,
    ) -> None:
        self._queue = queue
        self._files = files
        self._pipeline = pipeline
        self._poll_interval = poll_interval_seconds
        self._stale_after = stale_after_seconds
        self._concurrency = concurrency
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def process_next(self) -> bool:
        record = self._queue.claim_next()
        if record is None:
            return False
        self._process(record)
        return True

    def reclaim_stale(self) -> int:
        return self._queue.reclaim_stale(self._stale_after)

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(
                target=self._run, name=f"ingestion-worker-{index}", daemon=True
            )
            for index in range(self._concurrency)
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout_seconds: float = 30.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout_seconds)
        self._threads = []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.reclaim_stale()
                handled = self.process_next()
            except Exception:
                logger.exception("ingestion worker iteration failed")
                handled = False
            if not handled:
                self._stop.wait(self._poll_interval)

    def _process(self, record: DocumentRecord) -> None:
        try:
            if record.storage_key is None:
                self._queue.mark_failed(record.id, MISSING_FILE_KEY)
                return
            self._pipeline.reingest(
                record.id,
                self._files.get(record.storage_key),
                lambda stage: self._queue.set_stage(record.id, stage),
                record.title,
            )
        except AppError as error:
            self._queue.mark_failed(record.id, error.client_message)
        except Exception:
            logger.exception("document %s failed unexpectedly", record.id)
            self._queue.mark_failed(record.id, UNEXPECTED_FAILURE)
        else:
            self._queue.mark_ready(record.id)
