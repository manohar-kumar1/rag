from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.answering.generator import message_text
from app.answering.protocol import Question

REWRITE_INSTRUCTIONS = """Rewrite the follow-up question as one standalone search query.

Rules:
- Replace pronouns and references such as "it", "they" or "that policy" with what they mean \
in the conversation.
- Keep names, numbers, identifiers and quoted terms exactly as written.
- Do not answer the question and do not add facts.
- Reply with the query only, on one line."""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            REWRITE_INSTRUCTIONS.replace("{", "{{").replace("}", "}}")
            + "\n\nConversation:\n{conversation}\n\nFollow-up question: {question}",
        )
    ]
)


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
