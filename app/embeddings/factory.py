from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.embeddings.local import SentenceTransformerEmbedder
from app.embeddings.protocol import Embedder


def create_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(
        SentenceTransformer(settings.embedding_model),
        settings.embedding_dimension,
    )
