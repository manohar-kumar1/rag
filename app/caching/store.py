import time
from dataclasses import dataclass
from typing import Protocol

from psycopg.types.json import Jsonb

from app.answering.protocol import Answer
from app.caching.serialization import answer_from_json, answer_to_json
from app.db.source import ConnectionSource
from app.db.vector import to_vector_literal

LOOKUP = (
    "update answer_cache set hits = hits + 1 "
    "where scope = %s and question_key = %s and corpus_version = %s and expires_at > now() "
    "returning answer"
)
NEAREST = (
    "select question, 1 - (embedding operator(extensions.<=>) %(embedding)s::extensions.vector) "
    "from answer_cache where scope = %(scope)s and corpus_version = %(version)s "
    "and expires_at > now() and embedding is not null "
    "order by embedding operator(extensions.<=>) %(embedding)s::extensions.vector limit 1"
)
DELETE_EXPIRED = "delete from answer_cache where expires_at <= now()"
SAVE = (
    "insert into answer_cache "
    "(scope, question_key, corpus_version, question, embedding, answer, expires_at) "
    "values (%s, %s, %s, %s, %s::extensions.vector, %s, now() + make_interval(secs => %s)) "
    "on conflict (scope, question_key, corpus_version) do update set "
    "question = excluded.question, embedding = excluded.embedding, answer = excluded.answer, "
    "expires_at = excluded.expires_at"
)
CORPUS_VERSION = (
    "select count(*)::text || ':' || coalesce(max(updated_at)::text, '-') || ':' || "
    "coalesce((select max(id) from chunks)::text, '-') from documents"
)


@dataclass(frozen=True)
class CacheEntry:
    scope: str
    key: str
    version: str


@dataclass(frozen=True)
class NearMatch:
    question: str
    similarity: float


class AnswerCache(Protocol):
    def lookup(self, entry: CacheEntry) -> Answer | None: ...

    def nearest(self, entry: CacheEntry, embedding: list[float]) -> NearMatch | None: ...

    def save(
        self, entry: CacheEntry, question: str, embedding: list[float] | None, answer: Answer
    ) -> None: ...


class PostgresAnswerCache:
    def __init__(self, source: ConnectionSource, ttl_seconds: float) -> None:
        self._source = source
        self._ttl_seconds = ttl_seconds

    def lookup(self, entry: CacheEntry) -> Answer | None:
        with self._source.connection() as conn:
            row = conn.execute(LOOKUP, (entry.scope, entry.key, entry.version)).fetchone()
        return None if row is None else answer_from_json(row[0])

    def nearest(self, entry: CacheEntry, embedding: list[float]) -> NearMatch | None:
        params = {
            "embedding": to_vector_literal(embedding),
            "scope": entry.scope,
            "version": entry.version,
        }
        with self._source.connection() as conn:
            row = conn.execute(NEAREST, params).fetchone()
        return None if row is None else NearMatch(row[0], float(row[1]))

    def save(
        self, entry: CacheEntry, question: str, embedding: list[float] | None, answer: Answer
    ) -> None:
        vector = None if embedding is None else to_vector_literal(embedding)
        with self._source.connection() as conn, conn.transaction():
            conn.execute(DELETE_EXPIRED)
            conn.execute(
                SAVE,
                (
                    entry.scope,
                    entry.key,
                    entry.version,
                    question,
                    vector,
                    Jsonb(answer_to_json(answer)),
                    self._ttl_seconds,
                ),
            )


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = sum(a * a for a in left) ** 0.5 * sum(b * b for b in right) ** 0.5
    return dot / norms if norms else 0.0


class InMemoryAnswerCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[CacheEntry, tuple[float, str, list[float] | None, Answer]] = {}

    def lookup(self, entry: CacheEntry) -> Answer | None:
        stored = self._entries.get(entry)
        if stored is None or stored[0] <= time.monotonic():
            return None
        return stored[3]

    def nearest(self, entry: CacheEntry, embedding: list[float]) -> NearMatch | None:
        now = time.monotonic()
        candidates = [
            NearMatch(question, cosine(embedding, vector))
            for key, (expires, question, vector, _) in self._entries.items()
            if key.scope == entry.scope
            and key.version == entry.version
            and vector is not None
            and expires > now
        ]
        return max(candidates, key=lambda match: match.similarity, default=None)

    def save(
        self, entry: CacheEntry, question: str, embedding: list[float] | None, answer: Answer
    ) -> None:
        now = time.monotonic()
        self._entries = {
            key: stored for key, stored in self._entries.items() if stored[0] > now
        }
        self._entries[entry] = (now + self._ttl_seconds, question, embedding, answer)


class PostgresCorpusVersion:
    def __init__(self, source: ConnectionSource) -> None:
        self._source = source

    def current(self) -> str:
        with self._source.connection() as conn:
            return conn.execute(CORPUS_VERSION).fetchone()[0]
