from dataclasses import dataclass
from hashlib import sha256

from app.embeddings.protocol import Embedder
from app.exceptions.exception import EmbeddingError, EmbeddingFailed, EmptyDocumentError
from app.ingestion.chunking import Chunker
from app.ingestion.detect import FileType, detect_file_type
from app.ingestion.parsers import ParserRegistry
from app.ingestion.store import DocumentStore, NewDocument


@dataclass(frozen=True)
class IngestionResult:
    document_id: int
    created: bool
    chunk_count: int


@dataclass(frozen=True)
class PreparedContent:
    file_type: FileType
    chunks: list[str]
    vectors: list[list[float]]


class IngestionPipeline:
    def __init__(
        self,
        parsers: ParserRegistry,
        chunker: Chunker,
        embedder: Embedder,
        store: DocumentStore,
    ) -> None:
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest(
        self, data: bytes, source: str, title: str | None = None
    ) -> IngestionResult:
        content_hash = sha256(data).hexdigest()
        existing = self._store.find_by_hash(content_hash)
        if existing is not None:
            return IngestionResult(existing.id, False, existing.chunk_count)
        prepared = self._prepare(data)
        stored = self._store.add_document(
            NewDocument(
                source=source,
                title=title,
                content_hash=content_hash,
                metadata={"file_type": prepared.file_type.value, "size_bytes": len(data)},
            ),
            prepared.chunks,
            prepared.vectors,
        )
        return IngestionResult(stored.id, stored.created, stored.chunk_count)

    def reingest(self, document_id: int, data: bytes) -> IngestionResult:
        prepared = self._prepare(data)
        self._store.replace_chunks(
            document_id, sha256(data).hexdigest(), prepared.chunks, prepared.vectors
        )
        return IngestionResult(document_id, False, len(prepared.chunks))

    def _prepare(self, data: bytes) -> PreparedContent:
        file_type = detect_file_type(data)
        text = self._parsers.parse(data, file_type)
        chunks = self._chunker.split(text) if text.strip() else []
        if not chunks:
            raise EmptyDocumentError
        return PreparedContent(file_type, chunks, self._embed(chunks))

    def _embed(self, chunks: list[str]) -> list[list[float]]:
        try:
            vectors = self._embedder.embed_documents(chunks)
        except EmbeddingError as error:
            raise EmbeddingFailed(str(error)) from error
        if len(vectors) != len(chunks):
            raise EmbeddingFailed(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        return vectors
