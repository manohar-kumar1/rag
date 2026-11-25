from app.exceptions.errors import DocumentNotFound, JobNotFound
from app.ingestion.document_repository import DocumentRepository
from app.ingestion.records import DocumentRecord


class JobService:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def get(self, job_id: int) -> DocumentRecord:
        try:
            return self._repository.get(job_id)
        except DocumentNotFound as error:
            raise JobNotFound(job_id) from error
