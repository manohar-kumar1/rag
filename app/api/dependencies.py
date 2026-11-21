from typing import Annotated

from fastapi import Depends, Request

from app.answering.protocol import Answerer
from app.composition import Container
from app.core.readiness import ReadinessChecker
from app.ingestion.documents import DocumentService
from app.ingestion.idempotency import IdempotentUploads
from app.ingestion.jobs import JobService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_uploads(
    container: Annotated[Container, Depends(get_container)],
) -> IdempotentUploads:
    return container.uploads


def get_document_service(
    container: Annotated[Container, Depends(get_container)],
) -> DocumentService:
    return container.document_service


def get_job_service(
    container: Annotated[Container, Depends(get_container)],
) -> JobService:
    return container.jobs


def get_readiness(
    container: Annotated[Container, Depends(get_container)],
) -> ReadinessChecker:
    return container.readiness


def get_answerer(container: Annotated[Container, Depends(get_container)]) -> Answerer:
    return container.answerer


def get_job_retry_after(request: Request) -> int:
    return request.app.state.settings.job_retry_after_seconds
