import re
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import replace

from app.answering.protocol import Answer, Answerer, Question, StreamEvent, Turn


class SensitiveDataMasker:
    def __init__(self, patterns: Mapping[str, str]) -> None:
        self._patterns = [(name, re.compile(pattern)) for name, pattern in patterns.items()]

    def mask(self, text: str) -> str:
        for name, pattern in self._patterns:
            text = pattern.sub(f"[{name}]", text)
        return text

    def mask_question(self, question: Question) -> Question:
        return replace(
            question,
            text=self.mask(question.text),
            history=tuple(Turn(turn.role, self.mask(turn.content)) for turn in question.history),
        )


class MaskingAnswerer:
    def __init__(self, inner: Answerer, masker: SensitiveDataMasker) -> None:
        self._inner = inner
        self._masker = masker

    async def ask(self, question: Question) -> Answer:
        return await self._inner.ask(self._masker.mask_question(question))

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        async with aclosing(self._inner.stream(self._masker.mask_question(question))) as events:
            async for event in events:
                yield event
