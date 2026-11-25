from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    source: str
    title: str | None
    content_hash: str
    status: str
    error: str | None
    original_filename: str | None
    content_type: str | None
    size_bytes: int | None
    storage_key: str | None
    created_at: datetime
    processing_started_at: datetime | None
    processed_at: datetime | None
    processing_stage: str | None
    doc_type: str | None = None
    department: str | None = None
    product: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class NewUpload:
    source: str
    title: str | None
    content_hash: str
    original_filename: str
    content_type: str
    size_bytes: int
    storage_key: str
    doc_type: str | None = None
