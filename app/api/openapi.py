from fastapi import FastAPI

from app.exceptions.handlers import PROBLEM_MEDIA_TYPE

JSON_MEDIA_TYPE = "application/json"
FIRST_ERROR_STATUS = 400


def document_problem_media_type(app: FastAPI) -> None:
    build_schema = app.openapi

    def openapi() -> dict:
        schema = build_schema()
        for operations in schema.get("paths", {}).values():
            for operation in operations.values():
                for status, response in operation.get("responses", {}).items():
                    if status.isdigit() and int(status) >= FIRST_ERROR_STATUS:
                        content = response.get("content", {})
                        if JSON_MEDIA_TYPE in content:
                            content[PROBLEM_MEDIA_TYPE] = content.pop(JSON_MEDIA_TYPE)
        return schema

    app.openapi = openapi
