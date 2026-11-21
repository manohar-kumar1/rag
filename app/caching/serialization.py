import dataclasses
import re

from app.answering.citations import Citation
from app.answering.protocol import Answer
from app.retrieval.protocol import RetrievedChunk

WHITESPACE = re.compile(r"\s+")
TRAILING_PUNCTUATION = "?!.;: "


def question_key(text: str) -> str:
    return WHITESPACE.sub(" ", text.strip().lower()).rstrip(TRAILING_PUNCTUATION)


def answer_to_json(answer: Answer) -> dict:
    return {
        "text": answer.text,
        "answered": answer.answered,
        "sources": [dataclasses.asdict(chunk) for chunk in answer.sources],
        "citations": [
            {"marker": citation.marker, "chunk": dataclasses.asdict(citation.chunk)}
            for citation in answer.citations
        ],
        "route": answer.route,
        "search_query": answer.search_query,
        "unanswered_parts": list(answer.unanswered_parts),
        "grounded": answer.grounded,
    }


def answer_from_json(data: dict) -> Answer:
    return Answer(
        text=data["text"],
        answered=data["answered"],
        sources=[RetrievedChunk(**chunk) for chunk in data["sources"]],
        citations=[
            Citation(item["marker"], RetrievedChunk(**item["chunk"])) for item in data["citations"]
        ],
        route=data["route"],
        search_query=data["search_query"],
        unanswered_parts=tuple(data["unanswered_parts"]),
        grounded=data["grounded"],
    )
