from fastapi import FastAPI

from app.exceptions.handlers import PROBLEM_MEDIA_TYPE

FIRST_ERROR_STATUS = 400


def as_problem_content(content: dict) -> dict:
    if not content or PROBLEM_MEDIA_TYPE in content:
        return content
    return {PROBLEM_MEDIA_TYPE: next(iter(content.values()))}


def document_problem_media_type(app: FastAPI) -> None:
    build_schema = app.openapi

    def openapi() -> dict:
        schema = build_schema()
        for operations in schema.get("paths", {}).values():
            for operation in operations.values():
                for status, response in operation.get("responses", {}).items():
                    if status.isdigit() and int(status) >= FIRST_ERROR_STATUS:
                        response["content"] = as_problem_content(response.get("content", {}))
        return schema

    app.openapi = openapi
