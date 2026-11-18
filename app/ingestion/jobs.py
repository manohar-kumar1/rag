from app.exceptions.exception import DocumentNotFound, JobNotFound
from app.ingestion.records import DocumentRecord
from app.ingestion.repository import DocumentRepository


class JobService:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def get(self, job_id: int) -> DocumentRecord:
        try:
            return self._repository.get(job_id)
        except DocumentNotFound as error:
            raise JobNotFound(job_id) from error
