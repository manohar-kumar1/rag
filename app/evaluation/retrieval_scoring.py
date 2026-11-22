from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.retrieval.protocol import RetrievedChunk

REPORTED_CUTOFFS = (1, 3, 5, 8, 10)


class RetrievalCase(BaseModel):
    id: str
    question: str
    source: str
    pages: list[int] = Field(min_length=1)


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    first_hit_rank: int | None
    pages_found: int
    pages_expected: int
    returned: int


def covers_page(chunk: RetrievedChunk, case: RetrievalCase, page: int) -> bool:
    if chunk.source != case.source or chunk.page_start is None:
        return False
    return chunk.page_start <= page <= (chunk.page_end or chunk.page_start)


def score_case(case: RetrievalCase, chunks: Sequence[RetrievedChunk]) -> CaseScore:
    first_hit = next(
        (
            rank
            for rank, chunk in enumerate(chunks, start=1)
            if any(covers_page(chunk, case, page) for page in case.pages)
        ),
        None,
    )
    found = sum(
        any(covers_page(chunk, case, page) for chunk in chunks) for page in case.pages
    )
    return CaseScore(case.id, first_hit, found, len(case.pages), len(chunks))


def summarize(scores: Sequence[CaseScore]) -> dict:
    total = len(scores)
    returned = max((score.returned for score in scores), default=0)
    hit_at = {
        str(cutoff): round(
            sum(s.first_hit_rank is not None and s.first_hit_rank <= cutoff for s in scores)
            / total,
            4,
        )
        for cutoff in REPORTED_CUTOFFS
        if cutoff <= returned
    }
    reciprocal = sum(0 if s.first_hit_rank is None else 1 / s.first_hit_rank for s in scores)
    expected = sum(score.pages_expected for score in scores)
    return {
        "cases": total,
        "returned_per_question": returned,
        "hit_at": hit_at,
        "mrr": round(reciprocal / total, 4),
        "page_recall": round(sum(score.pages_found for score in scores) / expected, 4),
        "misses": [score.case_id for score in scores if score.first_hit_rank is None],
    }
