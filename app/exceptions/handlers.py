import logging
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions.exception import AppError

logger = logging.getLogger(__name__)


def error_body(code: str, message: str, details: list[dict] | None = None) -> dict:
    body = {"code": code, "message": message}
    if details:
        body["details"] = details
    return {"error": body}


def error_response(
    status_code: int, code: str, message: str, details: list[dict] | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, message, details))


async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
    if error.status_code >= 500:
        logger.error("%s: %s", error.code, error.message, exc_info=error)
    return error_response(error.status_code, error.code, error.client_message)


async def handle_http_exception(
    request: Request, error: StarletteHTTPException
) -> JSONResponse:
    return error_response(error.status_code, f"http_{error.status_code}", str(error.detail))


async def handle_validation_error(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    details = [
        {"location": [str(part) for part in item["loc"]], "message": item["msg"]}
        for item in error.errors()
    ]
    return error_response(422, "validation_error", "request validation failed", details)


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception("unhandled error", exc_info=error)
    return error_response(500, "internal_error", AppError.public_message)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)


def report_cli_error(error: AppError) -> int:
    print(f"error: {error.message}", file=sys.stderr)
    return 1
