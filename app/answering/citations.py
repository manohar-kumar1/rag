import re
from dataclasses import dataclass

from app.prompts import NO_ANSWER
from app.retrieval.protocol import RetrievedChunk

MARKER_GROUP = re.compile(r"\[(\d{1,4}(?:\s*,\s*\d{1,4})*)\]")


@dataclass(frozen=True)
class Citation:
    marker: int
    chunk: RetrievedChunk


def cited_markers(text: str, passage_count: int) -> list[int]:
    markers: dict[int, None] = {}
    for group in MARKER_GROUP.findall(text):
        for part in group.split(","):
            number = int(part)
            if 1 <= number <= passage_count:
                markers[number] = None
    return list(markers)


def cite(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    return [Citation(marker, chunks[marker - 1]) for marker in cited_markers(text, len(chunks))]


def is_no_answer(text: str) -> bool:
    return text.strip().strip(".").upper().startswith(NO_ANSWER)


class SentinelGate:
    def __init__(self, sentinel: str) -> None:
        self._sentinel = sentinel
        self._buffer = ""
        self._released = False
        self._suppressed = False

    @property
    def suppressed(self) -> bool:
        return self._suppressed

    def feed(self, delta: str) -> str:
        if self._released:
            return delta
        if self._suppressed:
            return ""
        self._buffer += delta
        probe = self._buffer.lstrip()
        if probe.upper().startswith(self._sentinel):
            self._suppressed = True
            return ""
        if self._sentinel.startswith(probe.upper()):
            return ""
        return self._release()

    def finish(self) -> str:
        if self._released or self._suppressed:
            return ""
        return self._release()

    def _release(self) -> str:
        self._released = True
        released, self._buffer = self._buffer, ""
        return released
