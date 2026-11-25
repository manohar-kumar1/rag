from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.answering.protocol import Question
from app.exceptions.errors import JudgementError
from app.llm.resilience import RetryPolicy, with_retries
from app.prompts import (
    GROUNDEDNESS_INSTRUCTIONS,
    ROUTE_INSTRUCTIONS,
    SUFFICIENCY_INSTRUCTIONS,
)
from app.query.protocol import Route


class RouteVerdict(BaseModel):
    route: Literal["simple", "complex"]


class ProbabilityVerdict(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)


def conversation(question: Question) -> list[BaseMessage]:
    messages: list[BaseMessage] = [
        HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content)
        for turn in question.history
    ]
    messages.append(HumanMessage(question.text))
    return messages


def passages_text(passages: Sequence[str]) -> str:
    return "\n\n".join(f"[{n}] {text}" for n, text in enumerate(passages, start=1))


def structured(model: BaseChatModel, schema: type[BaseModel], policy: RetryPolicy) -> Runnable:
    return with_retries(model.with_structured_output(schema, method="json_schema"), policy)


class LlmRouter:
    def __init__(self, model: BaseChatModel, policy: RetryPolicy) -> None:
        self._judge = structured(model, RouteVerdict, policy)

    def decide(self, question: Question) -> Route:
        verdict = self._judge.invoke([SystemMessage(ROUTE_INSTRUCTIONS), *conversation(question)])
        if not isinstance(verdict, RouteVerdict):
            raise JudgementError("router returned no usable route")
        return Route(verdict.route)


class LlmGrader:
    def __init__(self, model: BaseChatModel, policy: RetryPolicy) -> None:
        self._judge = structured(model, ProbabilityVerdict, policy)

    def sufficiency(self, question: str, passages: Sequence[str]) -> float:
        state = f"Question: {question}\n\nPassages:\n{passages_text(passages)}"
        return self._probability(SUFFICIENCY_INSTRUCTIONS, state)

    def groundedness(self, answer: str, passages: Sequence[str]) -> float:
        state = f"Answer: {answer}\n\nPassages:\n{passages_text(passages)}"
        return self._probability(GROUNDEDNESS_INSTRUCTIONS, state)

    def _probability(self, instructions: str, state: str) -> float:
        verdict = self._judge.invoke([SystemMessage(instructions), HumanMessage(state)])
        if not isinstance(verdict, ProbabilityVerdict):
            raise JudgementError("grader returned no verdict")
        return verdict.probability
