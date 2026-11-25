from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256

from app.embeddings.protocol import Embedder
from app.exceptions.errors import (
    EmbeddingError,
    EmbeddingFailed,
    EmptyDocumentError,
    FileTooLarge,
)
from app.ingestion.chunking import Chunker, Section
from app.ingestion.contextual import Contextualizer, NoContext
from app.ingestion.document_store import (
    ChunkText,
    DocumentStore,
    NewDocument,
    NewExternalDocument,
    SectionText,
    StoredContent,
)
from app.ingestion.external_sources import SourceDocument, SyncResult
from app.ingestion.file_types import MIME_TYPES, FileType, detect_file_type
from app.ingestion.layout import LayoutRegistry, ParsedDocument
from app.ingestion.status import IngestionStage

StageListener = Callable[[IngestionStage], None]


def ignore_stage(stage: IngestionStage) -> None:
    return None


@dataclass(frozen=True)
class IngestionResult:
    document_id: int
    created: bool
    chunk_count: int


def context_prefix(title: str | None, heading_path: str) -> str | None:
    parts = [part for part in (title, heading_path) if part]
    return " > ".join(parts) if parts else None


def join_context(prefix: str | None, generated: str | None) -> str | None:
    parts = [part for part in (prefix, generated) if part]
    return "\n".join(parts) if parts else None


class IngestionPipeline:
    def __init__(
        self,
        layouts: LayoutRegistry,
        chunker: Chunker,
        embedder: Embedder,
        embedding_key: str,
        store: DocumentStore,
        max_extracted_chars: int,
        contextualizer: Contextualizer | None = None,
    ) -> None:
        self._layouts = layouts
        self._chunker = chunker
        self._embedder = embedder
        self._embedding_key = embedding_key
        self._store = store
        self._max_extracted_chars = max_extracted_chars
        self._contextualizer = contextualizer or NoContext()

    def ingest(self, data: bytes, source: str, title: str | None = None) -> IngestionResult:
        content_hash = sha256(data).hexdigest()
        existing = self._store.find_by_hash(content_hash)
        if existing is not None:
            return IngestionResult(existing.id, False, existing.chunk_count)
        file_type = detect_file_type(data)
        content = self._prepare(self._layouts.parse(data, file_type), title or source, {})
        stored = self._store.add_document(
            NewDocument(
                source=source,
                title=title,
                content_hash=content_hash,
                metadata={},
                original_filename=source,
                content_type=MIME_TYPES[file_type],
                size_bytes=len(data),
                doc_type=file_type.value,
            ),
            content,
        )
        return IngestionResult(stored.id, stored.created, stored.chunk_count)

    def reingest(
        self,
        document_id: int,
        data: bytes,
        on_stage: StageListener = ignore_stage,
        title: str | None = None,
    ) -> IngestionResult:
        parsed = self._layouts.parse(data, detect_file_type(data))
        reusable = self._store.existing_vectors(document_id)
        content = self._prepare(parsed, title, reusable, on_stage)
        self._store.replace_chunks(document_id, sha256(data).hexdigest(), content)
        return IngestionResult(document_id, False, len(content.chunks))

    def ingest_source_document(self, document: SourceDocument) -> SyncResult:
        encoded = document.text.encode("utf-8")
        content_hash = sha256(encoded).hexdigest()
        existing = self._store.find_by_external_ref(document.reference)
        if existing is not None and existing.content_hash == content_hash:
            return SyncResult(document.reference, existing.id, "unchanged", existing.chunk_count)
        parsed = self._layouts.parse(encoded, FileType.TEXT)
        reusable = {} if existing is None else self._store.existing_vectors(existing.id)
        content = self._prepare(parsed, document.title, reusable)
        stored = self._store.upsert_external(
            NewExternalDocument(
                reference=document.reference,
                title=document.title,
                content_hash=content_hash,
                metadata=document.metadata,
                size_bytes=len(encoded),
            ),
            content,
        )
        return SyncResult(document.reference, stored.id, stored.outcome, stored.chunk_count)

    def _prepare(
        self,
        parsed: ParsedDocument,
        title: str | None,
        reusable: dict[str, list[float]],
        on_stage: StageListener = ignore_stage,
    ) -> StoredContent:
        if parsed.text_length > self._max_extracted_chars:
            raise FileTooLarge("extracted text is too long")
        sections = self._chunker.chunk(parsed) if parsed.text_length else []
        pieces = [chunk.text for section in sections for chunk in section.chunks]
        if not pieces:
            raise EmptyDocumentError
        generated = self._contextualizer.contextualize(
            "\n\n".join(section.text for section in sections), pieces
        )
        chunks = self._chunk_texts(sections, title, generated)
        on_stage(IngestionStage.EMBEDDING)
        vectors = self._vectors(chunks, reusable)
        return StoredContent(chunks, vectors, [self._section_text(s) for s in sections])

    def _chunk_texts(
        self, sections: Sequence[Section], title: str | None, generated: Sequence[str | None]
    ) -> list[ChunkText]:
        chunks: list[ChunkText] = []
        for section in sections:
            for chunk in section.chunks:
                context = join_context(
                    context_prefix(title, chunk.heading_path), generated[len(chunks)]
                )
                text = ChunkText(
                    content=chunk.text,
                    context=context,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading_path=chunk.heading_path,
                    token_count=chunk.token_count,
                    section_index=section.index,
                )
                chunks.append(replace(text, chunk_hash=self._hash(text)))
        return chunks

    def _hash(self, chunk: ChunkText) -> str:
        return sha256(f"{self._embedding_key}\n{chunk.embedding_text}".encode()).hexdigest()

    @staticmethod
    def _section_text(section: Section) -> SectionText:
        return SectionText(
            index=section.index,
            heading_path=section.heading_path,
            content=section.text,
            token_count=section.token_count,
            page_start=section.page_start,
            page_end=section.page_end,
        )

    def _vectors(
        self, chunks: Sequence[ChunkText], reusable: dict[str, list[float]]
    ) -> list[list[float]]:
        missing = [chunk for chunk in chunks if chunk.chunk_hash not in reusable]
        fresh = iter(self._embed([chunk.embedding_text for chunk in missing]))
        return [
            reusable[chunk.chunk_hash] if chunk.chunk_hash in reusable else next(fresh)
            for chunk in chunks
        ]

    def _embed(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        try:
            vectors = self._embedder.embed_documents(chunks)
        except EmbeddingError as error:
            raise EmbeddingFailed(str(error)) from error
        if len(vectors) != len(chunks):
            raise EmbeddingFailed(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        return vectors
