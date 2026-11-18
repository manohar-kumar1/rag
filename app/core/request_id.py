import re
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
STATE_KEY = "request_id"
ACCEPTED_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_id_of(scope: Scope) -> str | None:
    return scope.get("state", {}).get(STATE_KEY)


def choose_request_id(inbound: str | None) -> str:
    if inbound is not None and ACCEPTED_PATTERN.fullmatch(inbound):
        return inbound
    return uuid4().hex


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        inbound = dict(scope["headers"]).get(REQUEST_ID_HEADER.lower().encode())
        request_id = choose_request_id(inbound.decode("latin-1") if inbound else None)
        scope.setdefault("state", {})[STATE_KEY] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if REQUEST_ID_HEADER not in headers:
                    headers.append(REQUEST_ID_HEADER, request_id)
            await send(message)

        await self._app(scope, receive, send_with_id)
