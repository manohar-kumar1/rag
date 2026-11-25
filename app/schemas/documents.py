from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.status import DocumentStatus

MAX_METADATA_CHARS = 100


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
    doc_type: str | None
    department: str | None
    product: str | None
    version: str | None


class DocumentMetadataIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_type: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_CHARS)
    department: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_CHARS)
    product: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_CHARS)
    version: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_CHARS)


class DocumentPageOut(BaseModel):
    items: list[DocumentOut]
    next_cursor: str | None
