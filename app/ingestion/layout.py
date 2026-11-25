from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.exceptions.errors import UnsupportedFileType
from app.ingestion.file_types import FileType


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"


ATOMIC_KINDS = frozenset({BlockKind.TABLE, BlockKind.CODE})


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    page: int | None = None
    level: int = 0


@dataclass(frozen=True)
class ParsedDocument:
    blocks: tuple[Block, ...]

    @property
    def text_length(self) -> int:
        return sum(len(block.text) for block in self.blocks)


class LayoutParser(Protocol):
    def parse(self, data: bytes) -> ParsedDocument: ...


class LayoutRegistry:
    def __init__(self, parsers: dict[FileType, LayoutParser]) -> None:
        self._parsers = parsers

    def parse(self, data: bytes, file_type: FileType) -> ParsedDocument:
        parser = self._parsers.get(file_type)
        if parser is None:
            raise UnsupportedFileType(f"no parser for {file_type}")
        return parser.parse(data)
