from enum import StrEnum
from typing import Protocol

from app.answering.protocol import Question


class Route(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class RouteDecider(Protocol):
    def decide(self, question: Question) -> Route: ...
