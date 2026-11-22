from dataclasses import dataclass

from langchain_core.runnables import Runnable
from psycopg_pool import ConnectionPool

from app.answering.generator import AnswerGenerator
from app.answering.protocol import Answerer
from app.caching.answerer import CachingAnswerer
from app.caching.store import (
    AnswerCache,
    InMemoryAnswerCache,
    PostgresAnswerCache,
    PostgresCorpusVersion,
)
from app.conversations.store import PostgresConversationStore
from app.conversations.threads import ThreadedAnswerer
from app.core.config import Settings
from app.core.readiness import ReadinessChecker
from app.db.connection import create_pool
from app.db.probe import database_probe
from app.db.source import ConnectionSource
from app.embeddings.factory import create_embedder, create_jina_client
from app.embeddings.protocol import Embedder
from app.graph.answer_graph import (
    AgentPolicy,
    AnswerComponents,
    GraphAnswerer,
    build_answer_graph,
)
from app.guardrails.masking import MaskingAnswerer, SensitiveDataMasker
from app.ingestion.chunking import RecursiveChunker
from app.ingestion.contextual import Contextualizer, LlmContextualizer
from app.ingestion.documents import DocumentService
from app.ingestion.idempotency import IdempotentUploads, PostgresIdempotencyStore
from app.ingestion.jobs import JobService
from app.ingestion.parsers import default_parsers
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.repository import DocumentRepository, PostgresDocumentRepository
from app.ingestion.store import PostgresDocumentStore
from app.ingestion.sync import SourceSyncer
from app.ingestion.upload import UploadService
from app.ingestion.worker import IngestionWorker
from app.llm.factory import (
    create_chat_model,
    create_contextual_model,
    create_decision_model,
)
from app.observability.tracing import (
    TracedAnswerer,
    TracedRetriever,
    Tracing,
    create_tracing,
)
from app.retrieval.hybrid import PostgresHybridRetriever, RetrievalConfig
from app.retrieval.protocol import Retriever
from app.retrieval.rerank import JinaReranker, RerankingRetriever
from app.routing.decision import DecisionModelGrader, DecisionModelRouter
from app.routing.planning import LlmPlanner
from app.routing.rewrite import LlmQueryRewriter
from app.storage.factory import create_file_store
from app.storage.protocol import FileStore


@dataclass(frozen=True)
class Container:
    pool: ConnectionPool
    chat_model: Runnable
    embedder: Embedder
    ingestion: IngestionPipeline
    documents: DocumentRepository
    files: FileStore
    uploads: IdempotentUploads
    worker: IngestionWorker
    document_service: DocumentService
    jobs: JobService
    retriever: Retriever
    answerer: Answerer
    conversations: PostgresConversationStore
    readiness: ReadinessChecker
    syncer: SourceSyncer
    tracing: Tracing


def agent_policy(settings: Settings) -> AgentPolicy:
    return AgentPolicy(
        max_subquestions=settings.agent_max_subquestions,
        max_iterations=settings.agent_max_iterations,
        max_rewrites=settings.agent_max_rewrites,
        grade_threshold=settings.agent_grade_threshold,
        groundedness_enabled=settings.groundedness_check_enabled,
        groundedness_threshold=settings.groundedness_threshold,
    )


def build_retriever(
    settings: Settings, source: ConnectionSource, embedder: Embedder, tracing: Tracing
) -> Retriever:
    top_k = settings.rerank_candidates if settings.rerank_enabled else settings.retrieval_top_k
    retriever: Retriever = PostgresHybridRetriever(
        source,
        embedder,
        RetrievalConfig(
            rrf_k=settings.rrf_k,
            vector_weight=settings.retrieval_vector_weight,
            fts_weight=settings.retrieval_fts_weight,
            vector_candidates=settings.retrieval_vector_candidates,
            fts_candidates=settings.retrieval_fts_candidates,
            top_k=top_k,
        ),
    )
    if settings.rerank_enabled:
        reranker = JinaReranker(
            create_jina_client(settings),
            settings.rerank_model,
            settings.rerank_max_retries,
            settings.rerank_retry_backoff_seconds,
        )
        retriever = RerankingRetriever(retriever, reranker, settings.rerank_top_n)
    return TracedRetriever(retriever) if tracing.enabled else retriever


def build_answer_cache(settings: Settings, source: ConnectionSource) -> AnswerCache:
    if settings.answer_cache_store == "postgres":
        return PostgresAnswerCache(source, settings.answer_cache_ttl_seconds)
    return InMemoryAnswerCache(settings.answer_cache_ttl_seconds)


def build_answerer(
    settings: Settings,
    chat_model: Runnable,
    retriever: Retriever,
    tracing: Tracing,
    pool: ConnectionPool,
    embedder: Embedder,
    conversations: PostgresConversationStore,
) -> Answerer:
    decision_model = create_decision_model(settings)
    policy = agent_policy(settings)
    graph = build_answer_graph(
        AnswerComponents(
            decider=DecisionModelRouter(decision_model),
            rewriter=LlmQueryRewriter(chat_model),
            retriever=retriever,
            generator=AnswerGenerator(chat_model),
            planner=LlmPlanner(chat_model, settings.agent_max_subquestions),
            grader=DecisionModelGrader(decision_model),
        ),
        policy,
    )
    answerer: Answerer = TracedAnswerer(GraphAnswerer(graph, policy.recursion_limit), tracing)
    answerer = ThreadedAnswerer(answerer, conversations, settings.conversation_max_turns)
    if settings.answer_cache_enabled:
        answerer = CachingAnswerer(
            answerer,
            build_answer_cache(settings, pool),
            embedder,
            PostgresCorpusVersion(pool),
            settings.answer_cache_semantic_threshold,
        )
    return MaskingAnswerer(answerer, SensitiveDataMasker(settings.mask_patterns))


def build_contextualizer(settings: Settings) -> Contextualizer | None:
    if not settings.contextual_chunking_enabled:
        return None
    return LlmContextualizer(create_contextual_model(settings))


def build_container(settings: Settings, pool: ConnectionPool | None = None) -> Container:
    if pool is None:
        pool = create_pool(settings)
    embedder = create_embedder(settings)
    documents = PostgresDocumentRepository(pool)
    files = create_file_store(settings)
    chat_model = create_chat_model(settings)
    tracing = create_tracing(settings)
    retriever = build_retriever(settings, pool, embedder, tracing)
    conversations = PostgresConversationStore(pool)
    ingestion = IngestionPipeline(
        parsers=default_parsers(
            settings.docx_max_uncompressed_bytes, settings.max_pdf_pages
        ),
        chunker=RecursiveChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        store=PostgresDocumentStore(pool),
        max_extracted_chars=settings.max_extracted_chars,
        contextualizer=build_contextualizer(settings),
    )
    return Container(
        pool=pool,
        chat_model=chat_model,
        embedder=embedder,
        ingestion=ingestion,
        documents=documents,
        files=files,
        uploads=IdempotentUploads(
            UploadService(documents, files, settings.max_upload_bytes),
            PostgresIdempotencyStore(
                pool,
                settings.idempotency_ttl_seconds,
                settings.idempotency_lock_seconds,
            ),
            documents,
        ),
        document_service=DocumentService(documents, files),
        jobs=JobService(documents),
        retriever=retriever,
        answerer=build_answerer(
            settings, chat_model, retriever, tracing, pool, embedder, conversations
        ),
        conversations=conversations,
        tracing=tracing,
        readiness=ReadinessChecker(
            {
                "database": database_probe(pool, settings.db_pool_timeout_seconds),
                "storage": files.check,
            }
        ),
        syncer=SourceSyncer(ingestion),
        worker=IngestionWorker(
            queue=documents,
            files=files,
            pipeline=ingestion,
            poll_interval_seconds=settings.worker_poll_interval_seconds,
            stale_after_seconds=settings.worker_stale_after_seconds,
            concurrency=settings.worker_concurrency,
        ),
    )
