import re
from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.answering.generator import message_text

LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)](?!\d))\s*")

DECOMPOSE_INSTRUCTIONS = """Split the question into the separate sub-questions that must be \
looked up in a document collection to answer it.

Rules:
- Write at most {limit} sub-questions, one per line, with no numbering.
- Each sub-question must make sense on its own and keep names, numbers and identifiers exactly.
- If the question is already a single lookup, write it unchanged as the only line.
- Do not answer anything."""

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [("human", DECOMPOSE_INSTRUCTIONS + "\n\nQuestion: {question}")]
)

ALTERNATIVE_INSTRUCTIONS = """A document search did not find what this sub-question needs.

Write one different search query for the same information: use other words, synonyms or the \
likely heading of the section that would contain it. Keep names, numbers and identifiers \
exactly. Reply with the query only, on one line."""

ALTERNATIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            ALTERNATIVE_INSTRUCTIONS
            + "\n\nSub-question: {question}\nQueries already tried:\n{tried}",
        )
    ]
)


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
