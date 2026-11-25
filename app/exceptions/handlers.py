import logging
import sys
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from app.core.request_id import REQUEST_ID_HEADER, request_id_of
from app.exceptions.errors import AppError

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:rag:problem:"


def problem_title(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


def problem_body(
    scope: Scope,
    status_code: int,
    code: str,
    detail: str,
    errors: list[dict] | None = None,
) -> dict:
    request_id = request_id_of(scope)
    body: dict = {
        "type": f"{PROBLEM_TYPE_PREFIX}{code}",
        "title": problem_title(status_code),
        "status": status_code,
        "detail": detail,
        "instance": scope.get("path", ""),
        "code": code,
    }
    if request_id:
        body["request_id"] = request_id
    if errors:
        body["errors"] = errors
    return body


def app_error_body(scope: Scope, error: AppError) -> dict:
    if error.status_code >= 500:
        logger.error(
            "%s: %s (request %s)",
            error.code,
            error.message,
            request_id_of(scope),
            exc_info=error,
        )
    return problem_body(scope, error.status_code, error.code, error.client_message)


def unexpected_error_body(scope: Scope, error: Exception) -> dict:
    logger.exception("unhandled error (request %s)", request_id_of(scope), exc_info=error)
    return problem_body(scope, 500, "internal_error", AppError.public_message)


def problem_response(
    scope: Scope,
    status_code: int,
    code: str,
    detail: str,
    errors: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return problem_json(problem_body(scope, status_code, code, detail, errors), headers)


def problem_json(body: dict, headers: dict[str, str] | None = None) -> JSONResponse:
    request_id = body.get("request_id")
    response_headers = dict(headers or {})
    if request_id:
        response_headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        body,
        status_code=body["status"],
        headers=response_headers,
        media_type=PROBLEM_MEDIA_TYPE,
    )


async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
    return problem_json(app_error_body(request.scope, error))


async def handle_http_exception(
    request: Request, error: StarletteHTTPException
) -> JSONResponse:
    return problem_response(
        request.scope,
        error.status_code,
        f"http_{error.status_code}",
        str(error.detail),
        headers=error.headers,
    )


async def handle_validation_error(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    errors = [
        {"location": [str(part) for part in item["loc"]], "message": item["msg"]}
        for item in error.errors()
    ]
    return problem_response(
        request.scope, 422, "validation_error", "request validation failed", errors
    )


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    return problem_json(unexpected_error_body(request.scope, error))


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)


def report_cli_error(error: AppError) -> int:
    print(f"error: {error.message}", file=sys.stderr)
    return 1
