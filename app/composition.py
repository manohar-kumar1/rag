from dataclasses import dataclass

from langchain_core.runnables import Runnable
from psycopg_pool import ConnectionPool

from app.core.config import Settings
from app.db.connection import create_pool
from app.embeddings.factory import create_embedder
from app.embeddings.protocol import Embedder
from app.ingestion.chunking import RecursiveChunker
from app.ingestion.parsers import default_parsers
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store import PostgresDocumentStore
from app.llm.factory import create_chat_model


@dataclass(frozen=True)
class Container:
    pool: ConnectionPool
    chat_model: Runnable
    embedder: Embedder
    ingestion: IngestionPipeline


def build_container(settings: Settings) -> Container:
    pool = create_pool(settings)
    embedder = create_embedder(settings)
    return Container(
        pool=pool,
        chat_model=create_chat_model(settings),
        embedder=embedder,
        ingestion=IngestionPipeline(
            parsers=default_parsers(),
            chunker=RecursiveChunker(settings.chunk_size, settings.chunk_overlap),
            embedder=embedder,
            store=PostgresDocumentStore(pool),
        ),
    )
