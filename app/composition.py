from dataclasses import dataclass

from langchain_core.runnables import Runnable
from psycopg_pool import ConnectionPool

from app.core.config import Settings
from app.db.connection import create_pool
from app.embeddings.factory import create_embedder
from app.embeddings.protocol import Embedder
from app.llm.factory import create_chat_model


@dataclass(frozen=True)
class Container:
    pool: ConnectionPool
    chat_model: Runnable
    embedder: Embedder


def build_container(settings: Settings) -> Container:
    return Container(
        pool=create_pool(settings),
        chat_model=create_chat_model(settings),
        embedder=create_embedder(settings),
    )
