from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.exceptions.exception import InvalidSourceDocument

MAX_REFERENCE_LENGTH = 2048

SyncOutcome = Literal["created", "updated", "unchanged"]


@dataclass(frozen=True)
class SourceDocument:
    reference: str
    text: str
    title: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise InvalidSourceDocument("a source reference is required")
        if len(self.reference) > MAX_REFERENCE_LENGTH:
            raise InvalidSourceDocument("source reference is too long")


@dataclass(frozen=True)
class SyncResult:
    reference: str
    document_id: int
    outcome: SyncOutcome
    chunk_count: int


class Source(Protocol):
    def documents(self) -> Iterable[SourceDocument]: ...
