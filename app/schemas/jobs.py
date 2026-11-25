from datetime import datetime

from pydantic import BaseModel

from app.ingestion.records import DocumentRecord
from app.ingestion.status import JobStatus, job_status


class JobOut(BaseModel):
    id: int
    document_id: int
    status: JobStatus
    error: str | None
    created_at: datetime
    processed_at: datetime | None

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "JobOut":
        return cls(
            id=record.id,
            document_id=record.id,
            status=job_status(record),
            error=record.error,
            created_at=record.created_at,
            processed_at=record.processed_at,
        )


class UploadOut(JobOut):
    already_existed: bool

    @classmethod
    def from_result(cls, record: DocumentRecord, already_existed: bool) -> "UploadOut":
        return cls(
            **JobOut.from_record(record).model_dump(), already_existed=already_existed
        )
