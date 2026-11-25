from app.schemas.errors import ProblemOut

DESCRIPTIONS = {
    400: "The request is invalid, for example the file name is missing or the cursor is not valid",
    404: "The document or job does not exist",
    409: "The document is in the wrong state, or the idempotency key is still being processed",
    413: "The file exceeds the upload limit",
    415: "The file type is not supported or does not match its extension",
    422: "The request failed validation, or the idempotency key was used with a different file",
    502: "A backing service such as storage or embeddings failed",
    503: "A required dependency is unavailable",
}


def error_responses(*statuses: int) -> dict[int | str, dict]:
    return {
        status: {"model": ProblemOut, "description": DESCRIPTIONS[status]}
        for status in statuses
    }
