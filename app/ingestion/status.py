from enum import StrEnum
from typing import Protocol


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class IngestionStage(StrEnum):
    PARSING = "parsing"
    EMBEDDING = "embedding"


class JobStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


class StatusSource(Protocol):
    status: str
    processing_stage: str | None


def job_status(record: StatusSource) -> JobStatus:
    match DocumentStatus(record.status):
        case DocumentStatus.PENDING:
            return JobStatus.QUEUED
        case DocumentStatus.PROCESSING:
            if record.processing_stage == IngestionStage.EMBEDDING:
                return JobStatus.EMBEDDING
            return JobStatus.PARSING
        case DocumentStatus.READY:
            return JobStatus.DONE
        case DocumentStatus.FAILED:
            return JobStatus.FAILED
