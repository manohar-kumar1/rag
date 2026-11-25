from dataclasses import dataclass

from langchain_core.runnables import Runnable
from llama_cloud import LlamaCloud
from psycopg_pool import ConnectionPool

from app.answering.generator import AnswerGenerator
from app.answering.protocol import Answerer
from app.caching.answer_store import (
    AnswerCache,
    InMemoryAnswerCache,
    PostgresAnswerCache,
    PostgresCorpusVersion,
)
from app.caching.cached_answerer import CachingAnswerer
from app.conversations.store import PostgresConversationStore
from app.conversations.threaded_answerer import ThreadedAnswerer
from app.core.config import Settings
from app.core.readiness import ReadinessChecker
from app.db.connection_source import ConnectionSource
from app.db.pool import create_pool
from app.db.probe import database_probe
from app.embeddings.factory import create_embedder, create_jina_client
from app.embeddings.protocol import Embedder
from app.graph.answer_graph import (
    AgentPolicy,
    AnswerComponents,
    GraphAnswerer,
    build_answer_graph,
)
from app.guardrails.masking import MaskingAnswerer, SensitiveDataMasker
from app.ingestion.chunking import ChunkPlan, SectionChunker
from app.ingestion.contextual import Contextualizer, LlmContextualizer
from app.ingestion.document_repository import (
    DocumentRepository,
    PostgresDocumentRepository,
)
from app.ingestion.document_service import DocumentService
from app.ingestion.document_store import PostgresDocumentStore
from app.ingestion.idempotency import IdempotentUploads, PostgresIdempotencyStore
from app.ingestion.job_service import JobService
from app.ingestion.layout import LayoutParser, LayoutRegistry
from app.ingestion.llamaparse import LlamaParseLayoutParser
from app.ingestion.native_parsers import native_layouts
from app.ingestion.parse_cache import CachedLayoutParser, PostgresParseCache
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.source_sync import SourceSyncer
from app.ingestion.token_counter import CharacterTokenCounter
from app.ingestion.upload_service import UploadService
from app.ingestion.worker import IngestionWorker
from app.llm.factory import (
    create_base_model,
    create_chat_model,
    create_contextual_model,
    retry_policy,
)
from app.observability.tracing import (
    TracedAnswerer,
    TracedRetriever,
    Tracing,
    create_tracing,
)
from app.query.filter_extraction import (
    FilteringRetriever,
    LlmFilterExtractor,
    PostgresVocabularyReader,
)
from app.query.hyde import HydeEmbedder, HydeExpander
from app.query.llm_judges import LlmGrader, LlmRouter
from app.query.planning import LlmPlanner
from app.query.rewrite import LlmQueryRewriter
from app.retrieval.context_assembly import AssemblingRetriever, PostgresSectionReader
from app.retrieval.hybrid_search import PostgresHybridRetriever, RetrievalConfig
from app.retrieval.protocol import Retriever
from app.retrieval.rerank import JinaReranker, RerankingRetriever
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
    settings: Settings,
    source: ConnectionSource,
    embedder: Embedder,
    tracing: Tracing,
    chat_model: Runnable | None = None,
) -> Retriever:
    top_k = settings.rerank_candidates if settings.rerank_enabled else settings.retrieval_top_k
    query_embedder = HydeEmbedder(embedder, HydeExpander(chat_model)) if settings.hyde_enabled else embedder
    retriever: Retriever = PostgresHybridRetriever(
        source,
        query_embedder,
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
    if settings.parent_max_tokens > 0:
        retriever = AssemblingRetriever(
            retriever, PostgresSectionReader(source), settings.parent_max_tokens
        )
    if settings.filter_extraction_enabled:
        extractor = LlmFilterExtractor(chat_model, PostgresVocabularyReader(source))
        retriever = FilteringRetriever(retriever, extractor)
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
    judge_model = create_base_model(settings)
    llm_policy = retry_policy(settings)
    policy = agent_policy(settings)
    graph = build_answer_graph(
        AnswerComponents(
            decider=LlmRouter(judge_model, llm_policy),
            rewriter=LlmQueryRewriter(chat_model),
            retriever=retriever,
            generator=AnswerGenerator(chat_model),
            planner=LlmPlanner(chat_model, settings.agent_max_subquestions),
            grader=LlmGrader(judge_model, llm_policy),
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


def build_pdf_parser(settings: Settings, source: ConnectionSource) -> LayoutParser | None:
    if settings.pdf_parser != "llamaparse":
        return None
    client = LlamaCloud(
        api_key=settings.llamaparse_api_key,
        timeout=settings.llamaparse_timeout_seconds,
        max_retries=settings.llamaparse_max_retries,
    )
    parser = LlamaParseLayoutParser(
        client, settings.llamaparse_tier, settings.llamaparse_version, settings.max_pdf_pages
    )
    key = f"llamaparse:{settings.llamaparse_tier}:{settings.llamaparse_version}"
    return CachedLayoutParser(parser, PostgresParseCache(source), key)


def build_layouts(settings: Settings, source: ConnectionSource) -> LayoutRegistry:
    return native_layouts(
        settings.docx_max_uncompressed_bytes,
        settings.max_pdf_pages,
        build_pdf_parser(settings, source),
    )


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
    retriever = build_retriever(settings, pool, embedder, tracing, chat_model)
    conversations = PostgresConversationStore(pool)
    ingestion = IngestionPipeline(
        layouts=build_layouts(settings, pool),
        chunker=SectionChunker(
            ChunkPlan(
                settings.chunk_min_tokens,
                settings.chunk_max_tokens,
                settings.chunk_overlap_tokens,
            ),
            CharacterTokenCounter(),
        ),
        embedder=embedder,
        embedding_key=f"{settings.embedding_model}:{settings.embedding_dimension}",
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
