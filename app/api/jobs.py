from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse

from app.api.dependencies import get_job_retry_after, get_job_service
from app.api.errors import error_responses
from app.ingestion.jobs import JobService
from app.ingestion.records import DocumentRecord
from app.ingestion.status import JobStatus, job_status
from app.models.jobs import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


def job_headers(
    request: Request, record: DocumentRecord, retry_after: int
) -> dict[str, str]:
    headers = {"Location": str(request.app.url_path_for("get_job", job_id=record.id))}
    if not job_status(record).is_finished:
        headers["Retry-After"] = str(retry_after)
    return headers


@router.get(
    "/{job_id}",
    response_model=JobOut,
    summary="Follow an ingestion job",
    description=(
        "Returns the job status (queued, parsing, embedding, done or failed). While the job is "
        "unfinished the response carries a Retry-After header. Once it is done the response is "
        "a 303 redirect to the finished document."
    ),
    responses={
        303: {"description": "The job is done, Location points to the document"},
        **error_responses(404, 422),
    },
)
def get_job(
    job_id: int,
    request: Request,
    response: Response,
    jobs: Annotated[JobService, Depends(get_job_service)],
    retry_after: Annotated[int, Depends(get_job_retry_after)],
) -> JobOut | RedirectResponse:
    record = jobs.get(job_id)
    status = job_status(record)
    if status is JobStatus.DONE:
        location = request.app.url_path_for("get_document", document_id=record.id)
        return RedirectResponse(str(location), status_code=303)
    if not status.is_finished:
        response.headers["Retry-After"] = str(retry_after)
    return JobOut.from_record(record)
