import math
from typing import Protocol

CHARS_PER_TOKEN = 4


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharacterTokenCounter:
    def __init__(self, chars_per_token: int = CHARS_PER_TOKEN) -> None:
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self._chars_per_token))
