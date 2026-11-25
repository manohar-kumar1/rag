import logging
from collections.abc import Sequence

from langchain_core.runnables import Runnable

from app.answering.generator import message_text
from app.embeddings.protocol import Embedder
from app.prompts import HYDE_PROMPT
from app.retrieval.stage_log import log_stage

logger = logging.getLogger(__name__)



class HydeExpander:
    def __init__(self, model: Runnable) -> None:
        self._model = model

    def expand(self, question: str) -> str:
        return message_text(self._model.invoke(HYDE_PROMPT.format_messages(question=question)))


class HydeEmbedder:
    def __init__(self, inner: Embedder, expander: HydeExpander) -> None:
        self._inner = inner
        self._expander = expander

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            hypothetical = self._expander.expand(text)
        except Exception as error:  # noqa: BLE001
            logger.warning("hypothetical answer failed, embedding the question: %s", type(error).__name__)
            return self._inner.embed_query(text)
        log_stage("hyde", question=text, hypothetical_chars=len(hypothetical))
        return self._inner.embed_query(f"{text}\n\n{hypothetical}" if hypothetical else text)
