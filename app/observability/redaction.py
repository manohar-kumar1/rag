import dataclasses
import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[redacted]"
REASONING_BLOCK_TYPES = frozenset({"reasoning", "thinking", "redacted_thinking"})
SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|secret|secret[_-]?key|password|authorization|cookie)$",
    re.IGNORECASE,
)
MIN_SECRET_LENGTH = 8


class Redactor:
    def __init__(self, secrets: Iterable[str | None]) -> None:
        unique = {secret for secret in secrets if secret and len(secret) >= MIN_SECRET_LENGTH}
        self._secrets = sorted(unique, key=len, reverse=True)

    def __call__(self, payload: dict) -> dict:
        return self._clean(payload)

    def _clean(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        if isinstance(value, dict):
            return {
                key: REDACTED if SECRET_KEY.search(str(key)) else self._clean(item)
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._clean(item) for item in value if not is_reasoning_block(item)]
        if isinstance(value, str):
            return self._mask(value)
        return value

    def _mask(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text


def is_reasoning_block(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in REASONING_BLOCK_TYPES
