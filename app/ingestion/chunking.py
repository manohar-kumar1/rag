import re
from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.layout import ATOMIC_KINDS, Block, BlockKind, ParsedDocument
from app.ingestion.token_counter import TokenCounter

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
PATH_SEPARATOR = " > "
TABLE_HEADER_LINES = 2


@dataclass(frozen=True)
class ChunkPlan:
    min_tokens: int
    max_tokens: int
    overlap_tokens: int


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    token_count: int


@dataclass(frozen=True)
class Section:
    index: int
    heading_path: str
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int
    chunks: tuple[Chunk, ...]


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> list[Section]: ...


@dataclass
class Draft:
    path: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def has_body(self) -> bool:
        return any(block.kind is not BlockKind.HEADING for block in self.blocks)


@dataclass(frozen=True)
class Piece:
    text: str
    page: int | None
    atomic: bool


def pages_of(pieces: list[Piece]) -> tuple[int | None, int | None]:
    pages = [piece.page for piece in pieces if piece.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


class SectionChunker:
    def __init__(self, plan: ChunkPlan, counter: TokenCounter) -> None:
        self._plan = plan
        self._counter = counter

    def chunk(self, document: ParsedDocument) -> list[Section]:
        drafts = self._merge_small(self._merge_headings(self._drafts(document.blocks)))
        return [self._section(index, draft) for index, draft in enumerate(drafts)]

    @staticmethod
    def _drafts(blocks: tuple[Block, ...]) -> list[Draft]:
        stack: list[tuple[int, str]] = []
        drafts: list[Draft] = []
        current: Draft | None = None
        for block in blocks:
            if block.kind is BlockKind.HEADING:
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
                current = Draft(PATH_SEPARATOR.join(text for _, text in stack), [block])
                drafts.append(current)
            else:
                if current is None:
                    current = Draft("")
                    drafts.append(current)
                current.blocks.append(block)
        return drafts

    @staticmethod
    def _merge_headings(drafts: list[Draft]) -> list[Draft]:
        merged: list[Draft] = []
        carried: list[Block] = []
        for draft in drafts:
            if not draft.has_body:
                carried.extend(draft.blocks)
                continue
            draft.blocks = [*carried, *draft.blocks]
            carried = []
            merged.append(draft)
        if carried and merged:
            merged[-1].blocks.extend(carried)
        elif carried:
            merged.append(Draft("", carried))
        return merged

    def _merge_small(self, drafts: list[Draft]) -> list[Draft]:
        merged: list[Draft] = []
        for draft in drafts:
            tokens = self._tokens(draft.blocks)
            if (
                merged
                and tokens < self._plan.min_tokens
                and self._tokens(merged[-1].blocks) + tokens <= self._plan.max_tokens
            ):
                merged[-1].blocks.extend(draft.blocks)
            else:
                merged.append(draft)
        return merged

    def _tokens(self, blocks: list[Block]) -> int:
        return sum(self._counter.count(block.text) for block in blocks)

    def _section(self, index: int, draft: Draft) -> Section:
        pieces = [piece for block in draft.blocks for piece in self._pieces(block)]
        chunks = self._pack(pieces, draft.path)
        text = "\n\n".join(block.text for block in draft.blocks)
        page_start, page_end = pages_of(pieces)
        return Section(
            index, draft.path, text, page_start, page_end, self._counter.count(text), tuple(chunks)
        )

    def _pieces(self, block: Block) -> list[Piece]:
        atomic = block.kind in ATOMIC_KINDS
        if self._counter.count(block.text) <= self._plan.max_tokens:
            return [Piece(block.text, block.page, atomic)]
        if block.kind is BlockKind.TABLE:
            return [Piece(text, block.page, True) for text in self._table_parts(block.text)]
        if atomic:
            return [Piece(block.text, block.page, True)]
        return [Piece(text, block.page, False) for text in self._text_parts(block.text)]

    def _table_parts(self, text: str) -> list[str]:
        lines = text.splitlines()
        header, rows = lines[:TABLE_HEADER_LINES], lines[TABLE_HEADER_LINES:]
        parts: list[str] = []
        current: list[str] = []
        for row in rows:
            candidate = "\n".join([*header, *current, row])
            if current and self._counter.count(candidate) > self._plan.max_tokens:
                parts.append("\n".join([*header, *current]))
                current = []
            current.append(row)
        if current:
            parts.append("\n".join([*header, *current]))
        return parts or [text]

    def _text_parts(self, text: str) -> list[str]:
        parts: list[str] = []
        current = ""
        for sentence in self._sentences(text):
            joined = f"{current} {sentence}".strip()
            if current and self._counter.count(joined) > self._plan.max_tokens:
                parts.append(current)
                current = sentence
            else:
                current = joined
        if current:
            parts.append(current)
        return parts

    def _sentences(self, text: str) -> list[str]:
        sentences: list[str] = []
        for sentence in SENTENCE_END.split(text.strip()):
            words = sentence.split()
            while self._counter.count(" ".join(words)) > self._plan.max_tokens:
                cut = max(1, len(words) // 2)
                sentences.append(" ".join(words[:cut]))
                words = words[cut:]
            if words:
                sentences.append(" ".join(words))
        return sentences

    def _pack(self, pieces: list[Piece], path: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[Piece] = []
        used = 0
        fresh = 0

        def flush(carry_overlap: bool) -> None:
            nonlocal buffer, used, fresh
            text = "\n\n".join(piece.text for piece in buffer)
            start, end = pages_of(buffer)
            chunks.append(Chunk(text, path, start, end, self._counter.count(text)))
            last = buffer[-1]
            buffer = self._overlap(last) if carry_overlap and not last.atomic else []
            used = sum(self._counter.count(piece.text) for piece in buffer)
            fresh = 0

        for piece in pieces:
            tokens = self._counter.count(piece.text)
            if fresh and used + tokens > self._plan.max_tokens:
                flush(carry_overlap=not piece.atomic)
            buffer.append(piece)
            used += tokens
            fresh += 1
            if piece.atomic and tokens >= self._plan.max_tokens:
                flush(carry_overlap=False)
        if fresh:
            flush(carry_overlap=False)
        return chunks

    def _overlap(self, piece: Piece) -> list[Piece]:
        if self._plan.overlap_tokens <= 0:
            return []
        words = piece.text.split()
        tail: list[str] = []
        for word in reversed(words):
            if self._counter.count(" ".join([word, *tail])) > self._plan.overlap_tokens:
                break
            tail.insert(0, word)
        return [Piece(" ".join(tail), piece.page, False)] if tail and len(tail) < len(words) else []
