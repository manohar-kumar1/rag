from langchain_core.runnables import Runnable

from app.answering.generator import message_text
from app.answering.protocol import Question
from app.prompts import REWRITE_PROMPT


def format_history(question: Question) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in question.history)


class LlmQueryRewriter:
    def __init__(self, model: Runnable) -> None:
        self._model = model

    def rewrite(self, question: Question) -> str:
        if not question.history:
            return question.text
        messages = REWRITE_PROMPT.format_messages(
            conversation=format_history(question), question=question.text
        )
        rewritten = message_text(self._model.invoke(messages)).splitlines()
        return rewritten[0].strip() if rewritten and rewritten[0].strip() else question.text
