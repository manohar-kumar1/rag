import logging
from collections.abc import Sequence
from typing import Protocol

from langchain_core.runnables import Runnable

from app.answering.generator import message_text
from app.prompts import CONTEXT_PROMPT

logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHARS = 30000



class Contextualizer(Protocol):
    def contextualize(self, document: str, chunks: Sequence[str]) -> list[str | None]: ...


class NoContext:
    def contextualize(self, document: str, chunks: Sequence[str]) -> list[str | None]:
        return [None] * len(chunks)


def document_window(document: str, chunk: str) -> str:
    if len(document) <= MAX_DOCUMENT_CHARS:
        return document
    start = max(document.find(chunk), 0)
    begin = max(0, start - MAX_DOCUMENT_CHARS // 2)
    return document[begin : begin + MAX_DOCUMENT_CHARS]


class LlmContextualizer:
    def __init__(self, model: Runnable) -> None:
        self._model = model

    def contextualize(self, document: str, chunks: Sequence[str]) -> list[str | None]:
        return [self._context(document, chunk) for chunk in chunks]

    def _context(self, document: str, chunk: str) -> str | None:
        messages = CONTEXT_PROMPT.format_messages(
            document=document_window(document, chunk), chunk=chunk
        )
        try:
            context = message_text(self._model.invoke(messages))
        except Exception as error:  # noqa: BLE001
            logger.warning("chunk context failed, embedding without it: %s", type(error).__name__)
            return None
        return context or None
