from pydantic import BaseModel


class ProblemFieldError(BaseModel):
    location: list[str]
    message: str


class ProblemOut(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: str | None = None
    errors: list[ProblemFieldError] | None = None
