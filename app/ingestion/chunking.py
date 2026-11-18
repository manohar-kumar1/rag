from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker(Protocol):
    def split(self, text: str) -> list[str]: ...


class RecursiveChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> list[str]:
        return [chunk for chunk in self._splitter.split_text(text) if chunk.strip()]
