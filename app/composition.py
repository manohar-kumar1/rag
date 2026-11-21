from dataclasses import dataclass

from langchain_core.runnables import Runnable
from psycopg_pool import ConnectionPool

from app.core.config import Settings
from app.core.readiness import ReadinessChecker
from app.db.connection import create_pool
from app.db.probe import database_probe
from app.embeddings.factory import create_embedder
from app.embeddings.protocol import Embedder
from app.ingestion.chunking import RecursiveChunker
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
from app.llm.factory import create_chat_model
from app.retrieval.hybrid import PostgresHybridRetriever, RetrievalConfig
from app.retrieval.protocol import Retriever
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
    readiness: ReadinessChecker
    syncer: SourceSyncer


def build_container(settings: Settings) -> Container:
    pool = create_pool(settings)
    embedder = create_embedder(settings)
    documents = PostgresDocumentRepository(pool)
    files = create_file_store(settings)
    ingestion = IngestionPipeline(
        parsers=default_parsers(
            settings.docx_max_uncompressed_bytes, settings.max_pdf_pages
        ),
        chunker=RecursiveChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        store=PostgresDocumentStore(pool),
        max_extracted_chars=settings.max_extracted_chars,
    )
    return Container(
        pool=pool,
        chat_model=create_chat_model(settings),
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
        retriever=PostgresHybridRetriever(
            pool,
            embedder,
            RetrievalConfig(
                rrf_k=settings.rrf_k,
                vector_weight=settings.retrieval_vector_weight,
                fts_weight=settings.retrieval_fts_weight,
                vector_candidates=settings.retrieval_vector_candidates,
                fts_candidates=settings.retrieval_fts_candidates,
                top_k=settings.retrieval_top_k,
            ),
        ),
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
