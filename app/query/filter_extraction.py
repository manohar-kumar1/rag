import json
import logging
import re
import time
from collections.abc import Callable
from datetime import date
from typing import Protocol

from langchain_core.runnables import Runnable

from app.answering.generator import message_text
from app.db.connection_source import ConnectionSource
from app.prompts import FILTER_PROMPT
from app.retrieval.filters import EMPTY_FILTER, RetrievalFilter
from app.retrieval.protocol import RetrievedChunk, Retriever
from app.retrieval.stage_log import log_stage

logger = logging.getLogger(__name__)

TEXT_FIELDS = ("department", "product", "version", "doc_type")
VOCABULARY_CACHE_SECONDS = 60
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")



class VocabularyReader(Protocol):
    def values(self) -> dict[str, list[str]]: ...


class PostgresVocabularyReader:
    def __init__(
        self, source: ConnectionSource, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._source = source
        self._clock = clock
        self._cached: dict[str, list[str]] | None = None
        self._read_at = 0.0

    def values(self) -> dict[str, list[str]]:
        now = self._clock()
        if self._cached is None or now - self._read_at > VOCABULARY_CACHE_SECONDS:
            self._cached = self._read()
            self._read_at = now
        return self._cached

    def _read(self) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        with self._source.connection() as conn:
            for field in TEXT_FIELDS:
                rows = conn.execute(
                    f"select distinct {field} from documents where {field} is not null order by 1"
                ).fetchall()
                found[field] = [row[0] for row in rows]
        return found


def parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


class LlmFilterExtractor:
    def __init__(
        self, model: Runnable, vocabulary: VocabularyReader, today: Callable[[], date] = date.today
    ) -> None:
        self._model = model
        self._vocabulary = vocabulary
        self._today = today

    def extract(self, question: str) -> RetrievalFilter:
        vocabulary = self._vocabulary.values()
        if not any(vocabulary.values()):
            return EMPTY_FILTER
        listing = "\n".join(f"{field}: {', '.join(vocabulary[field]) or '(none)'}" for field in TEXT_FIELDS)
        messages = FILTER_PROMPT.format_messages(
            today=self._today().isoformat(), vocabulary=listing, question=question
        )
        raw = CODE_FENCE.sub("", message_text(self._model.invoke(messages)).strip())
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return EMPTY_FILTER
        if not isinstance(data, dict):
            return EMPTY_FILTER
        return RetrievalFilter(
            **{field: self._known(vocabulary[field], data.get(field)) for field in TEXT_FIELDS},
            updated_after=parse_date(data.get("updated_after")),
            updated_before=parse_date(data.get("updated_before")),
        )

    @staticmethod
    def _known(allowed: list[str], value: object) -> str | None:
        if not isinstance(value, str):
            return None
        lookup = {item.lower(): item for item in allowed}
        return lookup.get(value.strip().lower())


class FilterExtractor(Protocol):
    def extract(self, question: str) -> RetrievalFilter: ...


class FilteringRetriever:
    def __init__(self, inner: Retriever, extractor: FilterExtractor) -> None:
        self._inner = inner
        self._extractor = extractor

    def retrieve(
        self, query: str, filters: RetrievalFilter = EMPTY_FILTER
    ) -> list[RetrievedChunk]:
        if not filters.attributes_empty:
            return self._inner.retrieve(query, filters)
        try:
            extracted = self._extractor.extract(query)
        except Exception as error:  # noqa: BLE001
            logger.warning("filter extraction failed, searching without it: %s", type(error).__name__)
            extracted = EMPTY_FILTER
        merged = extracted.merged_under(filters)
        log_stage("filters", query=query, extracted=extracted.key, applied=merged.key)
        return self._inner.retrieve(query, merged)
