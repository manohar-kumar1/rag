from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.answering.protocol import Question, Turn
from app.query.protocol import Route


class EvalTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EvalCase(BaseModel):
    id: str
    question: str
    reference: str
    route: Route
    answerable: bool
    sources: list[str]
    history: list[EvalTurn] = Field(default_factory=list)
    follow_up: str | None = None

    def asked(self) -> Question:
        if self.follow_up is None:
            return Question(self.question)
        return Question(
            self.follow_up, tuple(Turn(turn.role, turn.content) for turn in self.history)
        )


def load_cases(path: Path) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]
