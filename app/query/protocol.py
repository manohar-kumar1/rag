from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from app.answering.protocol import Question


class Route(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class RouteDecider(Protocol):
    def decide(self, question: Question) -> Route: ...


class Grader(Protocol):
    def sufficiency(self, question: str, passages: Sequence[str]) -> float: ...

    def groundedness(self, answer: str, passages: Sequence[str]) -> float: ...
