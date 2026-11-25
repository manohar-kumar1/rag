import httpx

from app.core.config import Settings
from app.embeddings.jina import JinaEmbedder
from app.embeddings.protocol import Embedder
from app.embeddings.sentence_transformer import SentenceTransformerEmbedder

DIMENSION_PROBE_TEXT = "dimension check"


def create_local_embedder(settings: Settings) -> Embedder:
    # Imported here: the optional local-embeddings extra pulls in PyTorch, which deploys skip.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformerEmbedder(
        SentenceTransformer(settings.embedding_model),
        settings.embedding_dimension,
    )


def create_jina_client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        base_url=settings.jina_base_url,
        headers={"Authorization": f"Bearer {settings.jina_api_key}"},
        timeout=settings.jina_timeout_seconds,
    )


def create_jina_embedder(settings: Settings) -> Embedder:
    embedder = JinaEmbedder(
        create_jina_client(settings),
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.jina_batch_size,
        max_retries=settings.jina_max_retries,
        backoff_seconds=settings.jina_retry_backoff_seconds,
    )
    embedder.embed_query(DIMENSION_PROBE_TEXT)
    return embedder


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "jina":
        return create_jina_embedder(settings)
    return create_local_embedder(settings)
