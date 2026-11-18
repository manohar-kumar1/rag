from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import get_document_service, get_job_retry_after, get_uploads
from app.api.errors import error_responses
from app.api.jobs import job_headers
from app.api.pagination import MAX_CURSOR_LENGTH, decode_cursor, encode_cursor
from app.ingestion.documents import DocumentService
from app.ingestion.idempotency import IdempotentUploads
from app.models.documents import DocumentOut, DocumentPageOut
from app.models.jobs import JobOut, UploadOut

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_IDEMPOTENCY_KEY_LENGTH = 255
REPLAY_HEADER = "Idempotent-Replayed"

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    status_code=202,
    response_model=UploadOut,
    summary="Upload a document",
    description=(
        "Stores the file and queues it for ingestion. The response is 202 with a Location "
        "header pointing to the job to poll. If the same file was already uploaded the "
        "response is 200 with the existing job. Send an Idempotency-Key header to make a retry "
        "after a timeout safe: the same key with the same file replays the first response "
        f"({REPLAY_HEADER}: true), the same key with a different file is a 422, and a retry "
        "while the first request is still running is a 409."
    ),
    responses={
        200: {"description": "The same file was already uploaded", "model": UploadOut},
        **error_responses(400, 409, 413, 415, 422, 502),
    },
)
async def upload_document(
    file: UploadFile,
    request: Request,
    response: Response,
    uploads: Annotated[IdempotentUploads, Depends(get_uploads)],
    retry_after: Annotated[int, Depends(get_job_retry_after)],
    idempotency_key: Annotated[
        str | None, Header(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)
    ] = None,
) -> UploadOut:
    data = await file.read()
    result = await run_in_threadpool(
        uploads.submit, data, file.filename or "", idempotency_key
    )
    response.status_code = 202 if result.created else 200
    response.headers.update(job_headers(request, result.record, retry_after))
    if result.replayed:
        response.headers[REPLAY_HEADER] = "true"
    return UploadOut.from_result(result.record, already_existed=not result.created)


@router.get(
    "",
    response_model=DocumentPageOut,
    summary="List documents, newest first",
    description="Cursor paging: pass the next_cursor of one page as the cursor of the next.",
    responses=error_responses(400, 422),
)
def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_LENGTH)] = None,
) -> DocumentPageOut:
    page = service.list_page(limit, decode_cursor(cursor))
    next_cursor = (
        None if page.next_before_id is None else encode_cursor(page.next_before_id)
    )
    return DocumentPageOut(
        items=[DocumentOut.model_validate(item) for item in page.items],
        next_cursor=next_cursor,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Get a document and its ingestion status",
    responses=error_responses(404, 422),
)
def get_document(
    document_id: int,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentOut:
    return DocumentOut.model_validate(service.get(document_id))


@router.post(
    "/{document_id}/retry",
    status_code=202,
    response_model=JobOut,
    summary="Retry a failed document",
    description="Queues the failed document again. Follow the job named by the Location header.",
    responses=error_responses(404, 409, 422),
)
def retry_document(
    document_id: int,
    request: Request,
    response: Response,
    service: Annotated[DocumentService, Depends(get_document_service)],
    retry_after: Annotated[int, Depends(get_job_retry_after)],
) -> JobOut:
    record = service.retry(document_id)
    response.headers.update(job_headers(request, record, retry_after))
    return JobOut.from_record(record)


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Delete a document, its chunks and its stored original",
    responses=error_responses(404, 422),
)
def delete_document(
    document_id: int,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> Response:
    service.delete(document_id)
    return Response(status_code=204)
