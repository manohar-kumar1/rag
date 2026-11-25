from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, contextmanager

import langsmith
from langsmith import Client, traceable

from app.answering.protocol import Answer, Answerer, Question, StreamEvent
from app.core.config import Settings
from app.observability.redaction import Redactor
from app.retrieval.filters import EMPTY_FILTER, RetrievalFilter
from app.retrieval.protocol import RetrievedChunk, Retriever

TRACE_FLUSH_TIMEOUT_SECONDS = 10.0


class Tracing:
    def __init__(self, enabled: bool, client: Client | None, project: str | None) -> None:
        self.enabled = enabled
        self.client = client
        self.project = project

    @contextmanager
    def scope(self) -> Iterator[None]:
        with langsmith.tracing_context(
            enabled=self.enabled, client=self.client, project_name=self.project
        ):
            yield


def settings_secrets(settings: Settings) -> list[str | None]:
    return [
        settings.anthropic_api_key,
        settings.jina_api_key,
        settings.supabase_secret_key,
        settings.langsmith_api_key,
        settings.database_url,
    ]


def create_tracing(settings: Settings) -> Tracing:
    if not settings.langsmith_tracing:
        return Tracing(False, None, None)
    redact = Redactor(settings_secrets(settings))
    client = Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
        hide_inputs=redact,
        hide_outputs=redact,
        hide_metadata=redact,
    )
    return Tracing(True, client, settings.langsmith_project)


def retrieval_inputs(inputs: dict) -> dict:
    return {"query": inputs.get("query")}


def retrieval_outputs(chunks: list[RetrievedChunk]) -> dict:
    return {
        "documents": [
            {
                "page_content": chunk.content,
                "type": "Document",
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "score": chunk.score,
                    "vector_rank": chunk.vector_rank,
                    "fts_rank": chunk.fts_rank,
                },
            }
            for chunk in chunks
        ]
    }


class TracedRetriever:
    def __init__(self, inner: Retriever) -> None:
        self._inner = inner

    @traceable(
        run_type="retriever",
        name="hybrid_search",
        process_inputs=retrieval_inputs,
        process_outputs=retrieval_outputs,
    )
    def retrieve(
        self, query: str, filters: RetrievalFilter = EMPTY_FILTER
    ) -> list[RetrievedChunk]:
        return self._inner.retrieve(query, filters)


class TracedAnswerer:
    def __init__(self, inner: Answerer, tracing: Tracing) -> None:
        self._inner = inner
        self._tracing = tracing

    async def ask(self, question: Question) -> Answer:
        with self._tracing.scope():
            return await self._inner.ask(question)

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        with self._tracing.scope():
            async with aclosing(self._inner.stream(question)) as events:
                async for event in events:
                    yield event


def flush(tracing: Tracing) -> None:
    if tracing.client is not None:
        tracing.client.flush(timeout=TRACE_FLUSH_TIMEOUT_SECONDS)
