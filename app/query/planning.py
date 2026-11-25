import re
from collections.abc import Sequence

from langchain_core.runnables import Runnable

from app.answering.generator import message_text
from app.prompts import ALTERNATIVE_PROMPT, DECOMPOSE_PROMPT

LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)](?!\d))\s*")



def parse_lines(text: str, limit: int) -> list[str]:
    lines = [LIST_MARKER.sub("", line).strip() for line in text.splitlines()]
    unique = dict.fromkeys(line for line in lines if line)
    return list(unique)[:limit]


class LlmPlanner:
    def __init__(self, model: Runnable, max_subquestions: int) -> None:
        self._model = model
        self._max_subquestions = max_subquestions

    def decompose(self, question: str) -> list[str]:
        messages = DECOMPOSE_PROMPT.format_messages(
            question=question, limit=self._max_subquestions
        )
        return parse_lines(message_text(self._model.invoke(messages)), self._max_subquestions) or [
            question
        ]

    def alternative(self, question: str, tried: Sequence[str]) -> str:
        messages = ALTERNATIVE_PROMPT.format_messages(
            question=question, tried="\n".join(f"- {query}" for query in tried)
        )
        lines = parse_lines(message_text(self._model.invoke(messages)), 1)
        return lines[0] if lines else question
