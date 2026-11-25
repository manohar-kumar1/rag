from collections.abc import Sequence


def to_vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def from_vector_literal(text: str) -> list[float]:
    return [float(part) for part in text.strip("[]").split(",") if part]
