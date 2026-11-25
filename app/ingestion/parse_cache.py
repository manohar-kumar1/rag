from hashlib import sha256
from typing import Protocol

from psycopg.types.json import Jsonb

from app.db.connection_source import ConnectionSource
from app.ingestion.layout import Block, BlockKind, LayoutParser, ParsedDocument

LOOKUP = "select blocks from parse_cache where content_hash = %s and parser_key = %s"
STORE = (
    "insert into parse_cache (content_hash, parser_key, blocks) values (%s, %s, %s) "
    "on conflict (content_hash, parser_key) do update set blocks = excluded.blocks, "
    "created_at = now()"
)


def blocks_to_json(document: ParsedDocument) -> list[dict]:
    return [
        {"kind": block.kind.value, "text": block.text, "page": block.page, "level": block.level}
        for block in document.blocks
    ]


def blocks_from_json(rows: list[dict]) -> ParsedDocument:
    return ParsedDocument(
        tuple(Block(BlockKind(row["kind"]), row["text"], row["page"], row["level"]) for row in rows)
    )


class ParseCache(Protocol):
    def get(self, content_hash: str, parser_key: str) -> ParsedDocument | None: ...

    def put(self, content_hash: str, parser_key: str, document: ParsedDocument) -> None: ...


class PostgresParseCache:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def get(self, content_hash: str, parser_key: str) -> ParsedDocument | None:
        with self._source.connection() as conn:
            row = conn.execute(LOOKUP, (content_hash, parser_key)).fetchone()
        return None if row is None else blocks_from_json(row[0])

    def put(self, content_hash: str, parser_key: str, document: ParsedDocument) -> None:
        with self._source.connection() as conn:
            conn.execute(STORE, (content_hash, parser_key, Jsonb(blocks_to_json(document))))


class CachedLayoutParser:
    def __init__(self, inner: LayoutParser, cache: ParseCache, parser_key: str) -> None:
        self._inner = inner
        self._cache = cache
        self._parser_key = parser_key

    def parse(self, data: bytes) -> ParsedDocument:
        content_hash = sha256(data).hexdigest()
        cached = self._cache.get(content_hash, self._parser_key)
        if cached is not None:
            return cached
        document = self._inner.parse(data)
        self._cache.put(content_hash, self._parser_key, document)
        return document
