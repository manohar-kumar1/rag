from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.exceptions.handlers import problem_response

MULTIPART_OVERHEAD_BYTES = 64 * 1024
REJECTION_MESSAGE = "file exceeds the upload limit"


class UploadSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, limit_bytes: int, path: str) -> None:
        self._app = app
        self._limit = limit_bytes + MULTIPART_OVERHEAD_BYTES
        self._path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != self._path
        ):
            await self._app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self._limit:
            await self._reject(scope, receive, send)
            return

        received = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._limit:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not exceeded:
                await send(message)

        try:
            await self._app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = problem_response(scope, 413, "file_too_large", REJECTION_MESSAGE)
        await response(scope, receive, send)
