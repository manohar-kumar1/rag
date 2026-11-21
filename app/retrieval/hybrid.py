import re
from dataclasses import dataclass

from psycopg.rows import class_row

from app.db.schema import FULL_TEXT_LANGUAGE
from app.db.source import ConnectionSource
from app.db.vector import to_vector_literal
from app.embeddings.protocol import Embedder
from app.exceptions.exception import InvalidQuery
from app.retrieval.protocol import RetrievedChunk

DEFAULT_HNSW_EF_SEARCH = 40
MAX_QUERY_TERMS = 64
TERM_PATTERN = re.compile(r"[^\W_]+")


def to_or_query(text: str) -> str:
    terms = dict.fromkeys(TERM_PATTERN.findall(text.lower()))
    return " | ".join(list(terms)[:MAX_QUERY_TERMS])


HYBRID_SEARCH = f"""
with vector_hits as (
    select id, row_number() over (order by distance, id) as rank
    from (
        select id, embedding operator(extensions.<=>) %(embedding)s::extensions.vector as distance
        from chunks
        order by embedding operator(extensions.<=>) %(embedding)s::extensions.vector
        limit %(vector_candidates)s
    ) nearest
),
fts_hits as (
    select id, row_number() over (order by score desc, id) as rank
    from (
        select c.id, ts_rank_cd(c.fts, q.query) as score
        from chunks c, to_tsquery('{FULL_TEXT_LANGUAGE}', %(fts_query)s) as q(query)
        where c.fts @@ q.query
        order by score desc, c.id
        limit %(fts_candidates)s
    ) matching
),
fused as (
    select coalesce(v.id, f.id) as id,
           coalesce(%(vector_weight)s::float8 / (%(rrf_k)s + v.rank), 0)
         + coalesce(%(fts_weight)s::float8 / (%(rrf_k)s + f.rank), 0) as score,
           v.rank as vector_rank,
           f.rank as fts_rank
    from vector_hits v
    full outer join fts_hits f on f.id = v.id
)
select c.id as chunk_id, c.document_id, c.chunk_index, c.content, d.source, d.title,
       fused.score, fused.vector_rank, fused.fts_rank
from fused
join chunks c on c.id = fused.id
join documents d on d.id = c.document_id
order by fused.score desc, c.id
limit %(top_k)s
"""


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int
    vector_weight: float
    fts_weight: float
    vector_candidates: int
    fts_candidates: int
    top_k: int


class PostgresHybridRetriever:
    def __init__(
        self, source: ConnectionSource, embedder: Embedder, config: RetrievalConfig
    ) -> None:
        self._source = source
        self._embedder = embedder
        self._config = config

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        text = query.strip()
        if not text:
            raise InvalidQuery
        params = {
            "embedding": to_vector_literal(self._embedder.embed_query(text)),
            "fts_query": to_or_query(text),
            "rrf_k": self._config.rrf_k,
            "vector_weight": self._config.vector_weight,
            "fts_weight": self._config.fts_weight,
            "vector_candidates": self._config.vector_candidates,
            "fts_candidates": self._config.fts_candidates,
            "top_k": self._config.top_k,
        }
        ef_search = max(DEFAULT_HNSW_EF_SEARCH, self._config.vector_candidates)
        with (
            self._source.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=class_row(RetrievedChunk)) as cursor,
        ):
            # pgvector's HNSW scan returns at most ef_search rows, so cover every candidate
            conn.execute(
                "select set_config('hnsw.ef_search', %s, true)", (str(ef_search),)
            )
            return cursor.execute(HYBRID_SEARCH, params).fetchall()
