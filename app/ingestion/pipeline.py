from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from app.embeddings.protocol import Embedder
from app.exceptions.exception import (
    EmbeddingError,
    EmbeddingFailed,
    EmptyDocumentError,
    FileTooLarge,
)
from app.ingestion.chunking import Chunker
from app.ingestion.contextual import Contextualizer, NoContext
from app.ingestion.detect import MIME_TYPES, FileType, detect_file_type
from app.ingestion.parsers import ParserRegistry
from app.ingestion.sources import SourceDocument, SyncResult
from app.ingestion.status import IngestionStage
from app.ingestion.store import (
    ChunkText,
    DocumentStore,
    NewDocument,
    NewExternalDocument,
)

StageListener = Callable[[IngestionStage], None]


def ignore_stage(stage: IngestionStage) -> None:
    return None


@dataclass(frozen=True)
class IngestionResult:
    document_id: int
    created: bool
    chunk_count: int


@dataclass(frozen=True)
class PreparedContent:
    chunks: list[ChunkText]
    vectors: list[list[float]]


class IngestionPipeline:
    def __init__(
        self,
        parsers: ParserRegistry,
        chunker: Chunker,
        embedder: Embedder,
        store: DocumentStore,
        max_extracted_chars: int,
        contextualizer: Contextualizer | None = None,
    ) -> None:
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._max_extracted_chars = max_extracted_chars
        self._contextualizer = contextualizer or NoContext()

    def ingest(
        self, data: bytes, source: str, title: str | None = None
    ) -> IngestionResult:
        content_hash = sha256(data).hexdigest()
        existing = self._store.find_by_hash(content_hash)
        if existing is not None:
            return IngestionResult(existing.id, False, existing.chunk_count)
        file_type, prepared = self._prepare_file(data)
        stored = self._store.add_document(
            NewDocument(
                source=source,
                title=title,
                content_hash=content_hash,
                metadata={},
                original_filename=source,
                content_type=MIME_TYPES[file_type],
                size_bytes=len(data),
            ),
            prepared.chunks,
            prepared.vectors,
        )
        return IngestionResult(stored.id, stored.created, stored.chunk_count)

    def reingest(
        self, document_id: int, data: bytes, on_stage: StageListener = ignore_stage
    ) -> IngestionResult:
        _, prepared = self._prepare_file(data, on_stage)
        self._store.replace_chunks(
            document_id, sha256(data).hexdigest(), prepared.chunks, prepared.vectors
        )
        return IngestionResult(document_id, False, len(prepared.chunks))

    def ingest_source_document(self, document: SourceDocument) -> SyncResult:
        encoded = document.text.encode("utf-8")
        content_hash = sha256(encoded).hexdigest()
        existing = self._store.find_by_external_ref(document.reference)
        if existing is not None and existing.content_hash == content_hash:
            return SyncResult(
                document.reference, existing.id, "unchanged", existing.chunk_count
            )
        prepared = self._prepare_text(document.text)
        stored = self._store.upsert_external(
            NewExternalDocument(
                reference=document.reference,
                title=document.title,
                content_hash=content_hash,
                metadata=document.metadata,
                size_bytes=len(encoded),
            ),
            prepared.chunks,
            prepared.vectors,
        )
        return SyncResult(
            document.reference, stored.id, stored.outcome, stored.chunk_count
        )

    def _prepare_file(
        self, data: bytes, on_stage: StageListener = ignore_stage
    ) -> tuple[FileType, PreparedContent]:
        file_type = detect_file_type(data)
        return file_type, self._prepare_text(
            self._parsers.parse(data, file_type), on_stage
        )

    def _prepare_text(
        self, text: str, on_stage: StageListener = ignore_stage
    ) -> PreparedContent:
        if len(text) > self._max_extracted_chars:
            raise FileTooLarge("extracted text is too long")
        pieces = self._chunker.split(text) if text.strip() else []
        if not pieces:
            raise EmptyDocumentError
        contexts = self._contextualizer.contextualize(text, pieces)
        chunks = [ChunkText(piece, context) for piece, context in zip(pieces, contexts, strict=True)]
        on_stage(IngestionStage.EMBEDDING)
        return PreparedContent(chunks, self._embed([chunk.embedding_text for chunk in chunks]))

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
