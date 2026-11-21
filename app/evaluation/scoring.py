import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean

from app.answering.protocol import Answer, Answerer
from app.evaluation.dataset import EvalCase

RAGAS_METRICS = (
    "faithfulness",
    "llm_context_precision_with_reference",
    "context_recall",
    "answer_relevancy",
)


@dataclass
class CaseResult:
    case: EvalCase
    answer: Answer | None
    seconds: float
    error: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def retrieved_sources(self) -> list[str]:
        return [] if self.answer is None else [chunk.source for chunk in self.answer.sources]

    @property
    def contexts(self) -> list[str]:
        return [] if self.answer is None else [chunk.content for chunk in self.answer.sources]

    @property
    def source_recall(self) -> float | None:
        if not self.case.sources:
            return None
        found = set(self.retrieved_sources)
        return sum(source in found for source in self.case.sources) / len(self.case.sources)


async def answer_cases(answerer: Answerer, cases: Sequence[EvalCase]) -> list[CaseResult]:
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            answer = await answerer.ask(case.asked())
            results.append(CaseResult(case, answer, time.perf_counter() - started))
        except Exception as error:  # noqa: BLE001
            results.append(
                CaseResult(case, None, time.perf_counter() - started, type(error).__name__)
            )
    return results


def average(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(mean(present), 4) if present else None


def summarize(results: Sequence[CaseResult]) -> dict:
    answerable = [r for r in results if r.case.answerable]
    unanswerable = [r for r in results if not r.case.answerable]
    routed = [r for r in results if r.answer is not None and r.answer.route is not None]
    return {
        "cases": len(results),
        "errors": sum(r.error is not None for r in results),
        "metrics": {
            name: average([r.metrics.get(name) for r in answerable]) for name in RAGAS_METRICS
        },
        "answered_rate_on_answerable": average(
            [float(r.answer.answered) for r in answerable if r.answer is not None]
        ),
        "refusal_rate_on_unanswerable": average(
            [float(not r.answer.answered) for r in unanswerable if r.answer is not None]
        ),
        "source_recall": average([r.source_recall for r in answerable]),
        "routing_accuracy": average(
            [float(r.answer.route == r.case.route.value) for r in routed]
        ),
        "routing_by_label": {
            label: {
                "cases": sum(r.case.route.value == label for r in routed),
                "predicted_simple": sum(
                    r.case.route.value == label and r.answer.route == "simple" for r in routed
                ),
            }
            for label in ("simple", "complex")
        },
        "mean_seconds_per_question": average([r.seconds for r in results]),
    }


def case_row(result: CaseResult) -> dict:
    answer = result.answer
    return {
        "id": result.case.id,
        "question": result.case.asked().text,
        "label_route": result.case.route.value,
        "answerable": result.case.answerable,
        "predicted_route": None if answer is None else answer.route,
        "search_query": None if answer is None else answer.search_query,
        "answered": None if answer is None else answer.answered,
        "answer": None if answer is None else answer.text,
        "cited_sources": [] if answer is None else [c.chunk.source for c in answer.citations],
        "retrieved_sources": result.retrieved_sources,
        "source_recall": result.source_recall,
        "seconds": round(result.seconds, 2),
        "error": result.error,
        "metrics": result.metrics,
    }
