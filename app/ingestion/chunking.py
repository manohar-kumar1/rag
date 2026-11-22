from dataclasses import dataclass
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_BREAK = "\f"


@dataclass(frozen=True)
class TextSpan:
    text: str
    start: int


class Chunker(Protocol):
    def spans(self, text: str) -> list[TextSpan]: ...

    def split(self, text: str) -> list[str]: ...


class RecursiveChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

    def spans(self, text: str) -> list[TextSpan]:
        documents = self._splitter.create_documents([text])
        return [
            TextSpan(document.page_content, document.metadata["start_index"])
            for document in documents
            if document.page_content.strip()
        ]

    def split(self, text: str) -> list[str]:
        return [span.text for span in self.spans(text)]


def page_at(text: str, offset: int) -> int | None:
    if offset < 0 or PAGE_BREAK not in text:
        return None
    return text.count(PAGE_BREAK, 0, offset) + 1
