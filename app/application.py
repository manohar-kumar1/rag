import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as system_router
from app.api.openapi import document_problem_media_type
from app.api.routes import v1_router
from app.api.upload_limit import UploadSizeLimitMiddleware
from app.composition import Container, build_container
from app.core.config import Settings
from app.core.request_id import REQUEST_ID_HEADER, RequestIdMiddleware
from app.exceptions.handlers import register_exception_handlers
from app.ingestion.upload_service import EXTENSION_TYPES
from app.observability.tracing import flush

logger = logging.getLogger(__name__)

ALLOWED_METHODS = ["GET", "POST", "PATCH", "DELETE"]
EXPOSED_HEADERS = ["Location", "Retry-After", "Idempotent-Replayed", REQUEST_ID_HEADER]


def describe(settings: Settings) -> str:
    extensions = ", ".join(sorted(EXTENSION_TYPES))
    return (
        "Upload documents, follow their ingestion jobs and manage them. "
        f"Supported file extensions: {extensions} (PDF needs a text layer). "
        f"Maximum upload size: {settings.max_upload_bytes} bytes. "
        "Every error is an RFC 9457 problem document (application/problem+json) with a "
        "stable code member."
    )


def prune_conversations(container: Container, retention_seconds: float) -> None:
    try:
        removed = container.conversations.prune(retention_seconds)
    except Exception as error:  # noqa: BLE001
        logger.warning("conversation pruning failed: %s", type(error).__name__)
        return
    logger.info("pruned %s expired conversation turns", removed)


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        container = build_container(settings)
        container.pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
        app.state.container = container
        prune_conversations(container, settings.conversation_retention_seconds)
        if settings.worker_enabled:
            container.worker.start()
        try:
            yield
        finally:
            container.worker.stop()
            flush(container.tracing)
            container.pool.close()

    app = FastAPI(
        title="RAG",
        description=describe(settings),
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    register_exception_handlers(app)
    app.include_router(system_router)
    app.include_router(v1_router)
    app.add_middleware(
        UploadSizeLimitMiddleware,
        limit_bytes=settings.max_upload_bytes,
        path=str(app.url_path_for("upload_document")),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=ALLOWED_METHODS,
        allow_headers=["*"],
        expose_headers=EXPOSED_HEADERS,
    )
    app.add_middleware(RequestIdMiddleware)
    document_problem_media_type(app)
    return app
