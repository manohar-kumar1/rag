import re
from dataclasses import dataclass

import psycopg
from psycopg.rows import class_row
from psycopg_pool import PoolTimeout

from app.db.connection_source import ConnectionSource
from app.db.search_config import FULL_TEXT_LANGUAGE
from app.db.vector import to_vector_literal
from app.embeddings.protocol import Embedder
from app.exceptions.errors import DatabaseUnavailable, InvalidQuery
from app.retrieval.filters import EMPTY_FILTER, RetrievalFilter
from app.retrieval.protocol import RetrievedChunk
from app.retrieval.stage_log import log_stage

DEFAULT_HNSW_EF_SEARCH = 40
MAX_QUERY_TERMS = 64
TERM_PATTERN = re.compile(r"[^\W_]+")


def to_or_query(text: str) -> str:
    terms = dict.fromkeys(TERM_PATTERN.findall(text.lower()))
    return " | ".join(list(terms)[:MAX_QUERY_TERMS])


ALLOWED_DOCUMENTS = """
allowed as (
    select d.id from documents d
    where (cardinality(%(document_ids)s::bigint[]) = 0 or d.id = any(%(document_ids)s))
      and (%(department)s::text is null or d.department = %(department)s)
      and (%(product)s::text is null or d.product = %(product)s)
      and (%(version)s::text is null or d.version = %(version)s)
      and (%(doc_type)s::text is null or d.doc_type = %(doc_type)s)
      and (%(updated_after)s::date is null or d.updated_at >= %(updated_after)s)
      and (%(updated_before)s::date is null or d.updated_at < %(updated_before)s::date + 1)
),"""

HYBRID_SEARCH = f"""
with {ALLOWED_DOCUMENTS}
vector_hits as (
    select id, row_number() over (order by distance, id) as rank
    from (
        select id, embedding operator(extensions.<=>) %(embedding)s::extensions.vector as distance
        from chunks
        where document_id in (select id from allowed)
        order by embedding operator(extensions.<=>) %(embedding)s::extensions.vector
        limit %(vector_candidates)s
    ) nearest
),
fts_hits as (
    select id, row_number() over (order by score desc, id) as rank
    from (
        select c.id, ts_rank_cd(c.fts, q.query) as score
        from chunks c, to_tsquery('{FULL_TEXT_LANGUAGE}', %(fts_query)s) as q(query)
        where c.fts @@ q.query and c.document_id in (select id from allowed)
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
       c.page_start, c.page_end, c.heading_path, c.section_id, fused.score, fused.vector_rank, fused.fts_rank
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

    def retrieve(
        self, query: str, filters: RetrievalFilter = EMPTY_FILTER
    ) -> list[RetrievedChunk]:
        text = query.strip()
        if not text:
            raise InvalidQuery
        params = {
            "document_ids": list(filters.document_ids),
            "department": filters.department,
            "product": filters.product,
            "version": filters.version,
            "doc_type": filters.doc_type,
            "updated_after": filters.updated_after,
            "updated_before": filters.updated_before,
            "embedding": to_vector_literal(self._embedder.embed_query(text)),
            "fts_query": to_or_query(text),
            "rrf_k": self._config.rrf_k,
            "vector_weight": self._config.vector_weight,
            "fts_weight": self._config.fts_weight,
            "vector_candidates": self._config.vector_candidates,
            "fts_candidates": self._config.fts_candidates,
            "top_k": self._config.top_k,
        }
        try:
            results = self._search(params)
        except (psycopg.OperationalError, PoolTimeout) as error:
            raise DatabaseUnavailable from error
        log_stage(
            "search",
            query=text,
            filters=filters.key,
            candidates=[
                [chunk.chunk_id, chunk.vector_rank, chunk.fts_rank, round(chunk.score, 4)]
                for chunk in results
            ],
        )
        return results

    def _search(self, params: dict) -> list[RetrievedChunk]:
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
            conn.execute("select set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            return cursor.execute(HYBRID_SEARCH, params).fetchall()
