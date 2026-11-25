import logging
from collections.abc import AsyncIterator, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from app.answering.passages import format_context
from app.exceptions.errors import AppError, GenerationError
from app.prompts import ANSWER_PROMPT
from app.retrieval.protocol import RetrievedChunk

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, model: Runnable) -> None:
        self._model = model

    async def astream(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[str]:
        messages = ANSWER_PROMPT.format_messages(context=format_context(chunks), question=question)
        try:
            async for chunk in self._model.astream(messages, config):
                if chunk.text:
                    yield chunk.text
        except AppError:
            raise
        except Exception as error:
            raise generation_failed(error) from error


def generation_failed(error: Exception) -> GenerationError:
    logger.warning("answer generation failed: %s", type(error).__name__)
    return GenerationError(f"answer generation failed: {type(error).__name__}")


def message_text(message: BaseMessage) -> str:
    return message.text.strip()
