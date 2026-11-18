import zipfile
from io import BytesIO
from typing import Protocol

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.exceptions.exception import FileTooLarge, ParseError, UnsupportedFileType
from app.ingestion.detect import FileType


class Parser(Protocol):
    def parse(self, data: bytes) -> str: ...


class TextParser:
    def parse(self, data: bytes) -> str:
        return data.decode("utf-8-sig")


class PdfParser:
    def __init__(self, max_pages: int) -> None:
        self._max_pages = max_pages

    def parse(self, data: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(data))
            encrypted = reader.is_encrypted
            page_count = 0 if encrypted else len(reader.pages)
        except Exception as error:
            raise ParseError("PDF could not be read") from error
        if encrypted:
            raise ParseError("encrypted PDFs are not supported")
        if page_count > self._max_pages:
            raise FileTooLarge("PDF has too many pages")
        try:
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as error:
            raise ParseError("PDF could not be read") from error
        return "\n\n".join(page for page in pages if page.strip())


class DocxParser:
    def __init__(self, max_uncompressed_bytes: int) -> None:
        self._max_uncompressed_bytes = max_uncompressed_bytes

    def parse(self, data: bytes) -> str:
        self._check_expanded_size(data)
        try:
            document = Document(BytesIO(data))
            blocks = [
                self._block_text(child, document) for child in document.element.body
            ]
        except Exception as error:
            raise ParseError("DOCX could not be read") from error
        return "\n\n".join(block for block in blocks if block.strip())

    def _check_expanded_size(self, data: bytes) -> None:
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                expanded = sum(item.file_size for item in archive.infolist())
        except zipfile.BadZipFile as error:
            raise ParseError("DOCX could not be read") from error
        if expanded > self._max_uncompressed_bytes:
            raise FileTooLarge("document is too large once uncompressed")

    @staticmethod
    def _block_text(element, document) -> str:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            return Paragraph(element, document).text
        if tag == "tbl":
            table = Table(element, document)
            return "\n".join(
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in table.rows
            )
        return ""


class ParserRegistry:
    def __init__(self, parsers: dict[FileType, Parser]) -> None:
        self._parsers = parsers

    def parse(self, data: bytes, file_type: FileType) -> str:
        parser = self._parsers.get(file_type)
        if parser is None:
            raise UnsupportedFileType(f"no parser for {file_type}")
        return parser.parse(data)


def default_parsers(
    docx_max_uncompressed_bytes: int, pdf_max_pages: int
) -> ParserRegistry:
    return ParserRegistry(
        {
            FileType.PDF: PdfParser(pdf_max_pages),
            FileType.DOCX: DocxParser(docx_max_uncompressed_bytes),
            FileType.TEXT: TextParser(),
        }
    )
