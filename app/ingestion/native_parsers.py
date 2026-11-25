import re
import zipfile
from io import BytesIO

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.exceptions.errors import FileTooLarge, ParseError
from app.ingestion.file_types import FileType
from app.ingestion.layout import (
    Block,
    BlockKind,
    LayoutParser,
    LayoutRegistry,
    ParsedDocument,
)

PAGE_BREAK = "\f"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
BLANK_LINES = re.compile(r"\n\s*\n")
WORD_LIST_STYLES = ("list bullet", "list number", "list paragraph")


def table_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = [f"| {' | '.join(row)} |" for row in padded]
    lines.insert(1, f"| {' | '.join('---' for _ in range(width))} |")
    return "\n".join(lines)


def paragraph_blocks(text: str, page: int | None) -> list[Block]:
    return [
        Block(BlockKind.PARAGRAPH, part.strip(), page)
        for part in BLANK_LINES.split(text)
        if part.strip()
    ]


class MarkdownLayoutParser:
    def parse(self, data: bytes) -> ParsedDocument:
        text = data.decode("utf-8-sig")
        pages = text.split(PAGE_BREAK)
        paged = len(pages) > 1
        blocks: list[Block] = []
        for number, page_text in enumerate(pages, start=1):
            blocks.extend(self._blocks(page_text, number if paged else None))
        return ParsedDocument(tuple(blocks))

    def _blocks(self, text: str, page: int | None) -> list[Block]:
        blocks: list[Block] = []
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
            elif FENCE.match(line):
                index = self._take_fence(lines, index, page, blocks)
            elif match := HEADING.match(line):
                blocks.append(Block(BlockKind.HEADING, match[2], page, len(match[1])))
                index += 1
            elif TABLE_ROW.match(line):
                index = self._take_run(lines, index, TABLE_ROW, BlockKind.TABLE, page, blocks)
            elif LIST_ITEM.match(line):
                index = self._take_list(lines, index, page, blocks)
            else:
                index = self._take_paragraph(lines, index, page, blocks)
        return blocks

    @staticmethod
    def _take_fence(lines: list[str], start: int, page: int | None, blocks: list[Block]) -> int:
        end = start + 1
        while end < len(lines) and not FENCE.match(lines[end]):
            end += 1
        blocks.append(Block(BlockKind.CODE, "\n".join(lines[start + 1 : end]), page))
        return min(end + 1, len(lines))

    @staticmethod
    def _take_run(
        lines: list[str], start: int, pattern: re.Pattern, kind: BlockKind,
        page: int | None, blocks: list[Block],
    ) -> int:
        end = start
        while end < len(lines) and pattern.match(lines[end]):
            end += 1
        blocks.append(Block(kind, "\n".join(lines[start:end]).strip(), page))
        return end

    @staticmethod
    def _take_list(lines: list[str], start: int, page: int | None, blocks: list[Block]) -> int:
        end = start
        while end < len(lines) and lines[end].strip() and (
            LIST_ITEM.match(lines[end]) or lines[end].startswith((" ", "\t"))
        ):
            end += 1
        blocks.append(Block(BlockKind.LIST, "\n".join(lines[start:end]).strip(), page))
        return end

    @staticmethod
    def _take_paragraph(lines: list[str], start: int, page: int | None, blocks: list[Block]) -> int:
        end = start
        while (
            end < len(lines)
            and lines[end].strip()
            and not (
                HEADING.match(lines[end]) or FENCE.match(lines[end]) or TABLE_ROW.match(lines[end])
            )
        ):
            end += 1
        blocks.append(Block(BlockKind.PARAGRAPH, "\n".join(lines[start:end]).strip(), page))
        return end


def open_pdf(data: bytes, max_pages: int) -> PdfReader:
    try:
        reader = PdfReader(BytesIO(data))
        encrypted = reader.is_encrypted
        page_count = 0 if encrypted else len(reader.pages)
    except Exception as error:
        raise ParseError("PDF could not be read") from error
    if encrypted:
        raise ParseError("encrypted PDFs are not supported")
    if page_count > max_pages:
        raise FileTooLarge("PDF has too many pages")
    return reader


class PypdfLayoutParser:
    def __init__(self, max_pages: int) -> None:
        self._max_pages = max_pages

    def parse(self, data: bytes) -> ParsedDocument:
        reader = open_pdf(data, self._max_pages)
        try:
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as error:
            raise ParseError("PDF could not be read") from error
        blocks = [
            block for number, text in enumerate(pages, start=1)
            for block in paragraph_blocks(text, number)
        ]
        return ParsedDocument(tuple(blocks))


class DocxLayoutParser:
    def __init__(self, max_uncompressed_bytes: int) -> None:
        self._max_uncompressed_bytes = max_uncompressed_bytes

    def parse(self, data: bytes) -> ParsedDocument:
        self._check_expanded_size(data)
        try:
            document = Document(BytesIO(data))
            blocks = self._blocks(document)
        except Exception as error:
            raise ParseError("DOCX could not be read") from error
        return ParsedDocument(tuple(blocks))

    def _check_expanded_size(self, data: bytes) -> None:
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                expanded = sum(item.file_size for item in archive.infolist())
        except zipfile.BadZipFile as error:
            raise ParseError("DOCX could not be read") from error
        if expanded > self._max_uncompressed_bytes:
            raise FileTooLarge("document is too large once uncompressed")

    def _blocks(self, document) -> list[Block]:
        blocks: list[Block] = []
        for element in document.element.body:
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "p":
                self._add_paragraph(Paragraph(element, document), blocks)
            elif tag == "tbl":
                rows = [[cell.text.strip() for cell in row.cells] for row in Table(element, document).rows]
                if rows:
                    blocks.append(Block(BlockKind.TABLE, table_markdown(rows)))
        return blocks

    @staticmethod
    def _add_paragraph(paragraph: Paragraph, blocks: list[Block]) -> None:
        text = paragraph.text.strip()
        if not text:
            return
        style = (paragraph.style.name if paragraph.style is not None else "").lower()
        if style == "title":
            blocks.append(Block(BlockKind.HEADING, text, None, 1))
        elif style.startswith("heading"):
            digits = "".join(ch for ch in style if ch.isdigit())
            blocks.append(Block(BlockKind.HEADING, text, None, int(digits) if digits else 1))
        elif style.startswith(WORD_LIST_STYLES):
            item = f"- {text}"
            if blocks and blocks[-1].kind is BlockKind.LIST:
                blocks[-1] = Block(BlockKind.LIST, f"{blocks[-1].text}\n{item}")
            else:
                blocks.append(Block(BlockKind.LIST, item))
        else:
            blocks.append(Block(BlockKind.PARAGRAPH, text))


def native_layouts(
    docx_max_uncompressed_bytes: int, pdf_max_pages: int, pdf_parser: LayoutParser | None = None
) -> LayoutRegistry:
    return LayoutRegistry(
        {
            FileType.PDF: pdf_parser or PypdfLayoutParser(pdf_max_pages),
            FileType.DOCX: DocxLayoutParser(docx_max_uncompressed_bytes),
            FileType.TEXT: MarkdownLayoutParser(),
        }
    )
