from io import BytesIO
from typing import Protocol

from pypdf import PdfReader

from app.exceptions.exception import ParseError, UnsupportedFileType
from app.ingestion.detect import FileType


class Parser(Protocol):
    def parse(self, data: bytes) -> str: ...


class TextParser:
    def parse(self, data: bytes) -> str:
        return data.decode("utf-8-sig")


class PdfParser:
    def parse(self, data: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(data))
            encrypted = reader.is_encrypted
            pages = [] if encrypted else [page.extract_text() or "" for page in reader.pages]
        except Exception as error:
            raise ParseError("PDF could not be read") from error
        if encrypted:
            raise ParseError("encrypted PDFs are not supported")
        return "\n\n".join(page for page in pages if page.strip())


class ParserRegistry:
    def __init__(self, parsers: dict[FileType, Parser]) -> None:
        self._parsers = parsers

    def parse(self, data: bytes, file_type: FileType) -> str:
        parser = self._parsers.get(file_type)
        if parser is None:
            raise UnsupportedFileType(f"no parser for {file_type}")
        return parser.parse(data)


def default_parsers() -> ParserRegistry:
    return ParserRegistry({FileType.PDF: PdfParser(), FileType.TEXT: TextParser()})
