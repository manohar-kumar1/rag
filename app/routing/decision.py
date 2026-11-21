from collections.abc import Sequence

from langchain_core.decisions import Choice, Noul
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from app.answering.protocol import Question
from app.exceptions.exception import DecisionError
from app.routing.protocol import Route

ROUTE_QUESTION_ID = "route"
ROUTE_INSTRUCTIONS = (
    "Classify the last user question in this conversation by how much work it takes to "
    "answer from a document collection."
)
ROUTE_CRITERIA = {
    Route.SIMPLE.value: (
        "One fact or one passage answers it: a definition, a value, a date, a single "
        "procedure step, or a direct lookup."
    ),
    Route.COMPLEX.value: (
        "It needs several passages or reasoning across them: comparisons, multi-part "
        "questions, aggregations, timelines, or questions whose parts depend on each other."
    ),
}


def conversation(question: Question) -> list[BaseMessage]:
    messages: list[BaseMessage] = [
        HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content)
        for turn in question.history
    ]
    messages.append(HumanMessage(question.text))
    return messages


YES_QUESTION_ID = "verdict"
SUFFICIENCY_INSTRUCTIONS = (
    "Do the passages contain the information needed to answer the question? Answer yes "
    "only if the question can be answered from the passages alone."
)
GROUNDEDNESS_INSTRUCTIONS = (
    "Is every factual claim in the answer stated in, or directly implied by, the passages? "
    "Answer no if any claim, number or name in the answer is missing from the passages."
)


def passages_text(passages: Sequence[str]) -> str:
    return "\n\n".join(f"[{n}] {text}" for n, text in enumerate(passages, start=1))


def yes_probability(classifier: Runnable, state: str, instructions: str) -> float:
    response = classifier.invoke(
        {"state": state, "questions": {YES_QUESTION_ID: Noul(instructions=instructions)}}
    )
    answer = response.nouls.get(YES_QUESTION_ID)
    if answer is None:
        raise DecisionError("decision model returned no verdict")
    return answer.noul


class DecisionModelGrader:
    def __init__(self, classifier: Runnable) -> None:
        self._classifier = classifier

    def sufficiency(self, question: str, passages: Sequence[str]) -> float:
        state = f"Question: {question}\n\nPassages:\n{passages_text(passages)}"
        return yes_probability(self._classifier, state, SUFFICIENCY_INSTRUCTIONS)

    def groundedness(self, answer: str, passages: Sequence[str]) -> float:
        state = f"Answer: {answer}\n\nPassages:\n{passages_text(passages)}"
        return yes_probability(self._classifier, state, GROUNDEDNESS_INSTRUCTIONS)


class DecisionModelRouter:
    def __init__(self, classifier: Runnable) -> None:
        self._classifier = classifier

    def decide(self, question: Question) -> Route:
        response = self._classifier.invoke(
            {
                "state": conversation(question),
                "questions": {
                    ROUTE_QUESTION_ID: Choice(
                        instructions=ROUTE_INSTRUCTIONS, criteria=ROUTE_CRITERIA
                    )
                },
            }
        )
        answer = response.choices.get(ROUTE_QUESTION_ID)
        if answer is None or answer.choice not in ROUTE_CRITERIA:
            raise DecisionError("decision model returned no usable route")
        return Route(answer.choice)
