from collections.abc import AsyncIterable
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.answering.protocol import (
    Answerer,
    DoneEvent,
    ProgressEvent,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from app.api.dependencies import get_answerer
from app.api.errors import error_responses
from app.exceptions.exception import AppError
from app.exceptions.handlers import app_error_body, unexpected_error_body
from app.models.query import ProgressOut, QueryIn, QueryOut, SourcesOut, TokenOut

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryOut,
    summary="Ask a question and get a cited answer",
    description=(
        "Answers from the uploaded documents only. `answered` is false when the documents do "
        "not contain the answer. Citations list the passages the answer refers to by marker."
    ),
    responses=error_responses(422, 502, 503),
)
async def query(
    body: QueryIn, answerer: Annotated[Answerer, Depends(get_answerer)]
) -> QueryOut:
    return QueryOut.from_answer(await answerer.ask(body.to_question()))


@router.post(
    "/stream",
    response_class=EventSourceResponse,
    summary="Ask a question and stream the answer as server-sent events",
    description=(
        "Event order: `progress` events as each step starts and ends, `sources` (the passages, numbered by marker), then `token` events with "
        "answer text, then `done` with the final answer and citations. A failure after the "
        "stream started is sent as an `error` event holding a problem document."
    ),
    responses=error_responses(422),
)
async def query_stream(
    body: QueryIn,
    request: Request,
    answerer: Annotated[Answerer, Depends(get_answerer)],
) -> AsyncIterable[ServerSentEvent]:
    try:
        async with aclosing(answerer.stream(body.to_question())) as events:
            async for event in events:
                if await request.is_disconnected():
                    break
                yield to_server_sent_event(event)
    except AppError as error:
        yield ServerSentEvent(event="error", data=app_error_body(request.scope, error))
    except Exception as error:  # noqa: BLE001
        yield ServerSentEvent(event="error", data=unexpected_error_body(request.scope, error))


def to_server_sent_event(event: StreamEvent) -> ServerSentEvent:
    match event:
        case SourcesEvent(sources):
            return ServerSentEvent(event="sources", data=SourcesOut.from_chunks(sources))
        case TokenEvent(text):
            return ServerSentEvent(event="token", data=TokenOut(text=text))
        case ProgressEvent(step, status):
            return ServerSentEvent(event="progress", data=ProgressOut(step=step, status=status))
        case DoneEvent(answer):
            return ServerSentEvent(event="done", data=QueryOut.from_answer(answer))
