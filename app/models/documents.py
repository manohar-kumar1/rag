from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.ingestion.status import DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    status: DocumentStatus
    error: str | None
    original_filename: str | None
    content_type: str | None
    size_bytes: int | None
    created_at: datetime
    processed_at: datetime | None


class DocumentPageOut(BaseModel):
    items: list[DocumentOut]
    next_cursor: str | None
