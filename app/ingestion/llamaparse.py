from collections.abc import Sequence
from typing import Any, Protocol

import llama_cloud

from app.exceptions.errors import LayoutServiceError, ParseError
from app.ingestion.layout import Block, BlockKind, ParsedDocument
from app.ingestion.native_parsers import open_pdf

UPLOAD_NAME = "document.pdf"
SERVICE_ERRORS = (
    llama_cloud.LlamaCloudError,
    llama_cloud.PollingError,
    llama_cloud.PollingTimeoutError,
)


class ParsingClient(Protocol):
    parsing: Any


def clean_heading(text: str) -> str:
    return text.strip().lstrip("#").strip().strip("*_ ").strip()


def item_block(item: Any, page: int) -> Block | None:
    kind = getattr(item, "type", None)
    if kind == "heading":
        text = clean_heading(item.value or item.md)
        return Block(BlockKind.HEADING, text, page, int(item.level or 1)) if text else None
    if kind == "text":
        return Block(BlockKind.PARAGRAPH, item.value.strip(), page) if item.value.strip() else None
    if kind == "list":
        return Block(BlockKind.LIST, item.md.strip(), page) if item.md.strip() else None
    if kind == "table":
        return Block(BlockKind.TABLE, item.md.strip(), page) if item.md.strip() else None
    if kind == "code":
        return Block(BlockKind.CODE, item.value.strip("\n"), page) if item.value.strip() else None
    if kind == "image" and item.caption.strip():
        return Block(BlockKind.PARAGRAPH, item.caption.strip(), page)
    return None


def blocks_from_pages(pages: Sequence[Any]) -> list[Block]:
    blocks: list[Block] = []
    for page in pages:
        if not getattr(page, "success", False):
            raise ParseError("the parsing service could not read every page")
        for item in page.items:
            if getattr(item, "merged_into_page", None) is not None:
                continue
            block = item_block(item, int(page.page_number))
            if block is not None:
                blocks.append(block)
    return blocks


class LlamaParseLayoutParser:
    def __init__(self, client: ParsingClient, tier: str, version: str, max_pages: int) -> None:
        self._client = client
        self._tier = tier
        self._version = version
        self._max_pages = max_pages

    def parse(self, data: bytes) -> ParsedDocument:
        open_pdf(data, self._max_pages)
        try:
            result = self._client.parsing.parse(
                tier=self._tier,
                version=self._version,
                upload_file=(UPLOAD_NAME, data),
                expand=["items"],
            )
        except SERVICE_ERRORS as error:
            raise LayoutServiceError(f"layout parse failed: {type(error).__name__}") from error
        if result.items is None:
            raise LayoutServiceError("layout parse returned no items")
        return ParsedDocument(tuple(blocks_from_pages(result.items.pages)))
