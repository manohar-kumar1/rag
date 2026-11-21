import math
from collections.abc import Sequence

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from app.embeddings.protocol import Embedder
from app.evaluation.scoring import RAGAS_METRICS, CaseResult
from app.exceptions.exception import JudgeUnavailable

JUDGE_TIMEOUT_SECONDS = 180
JUDGE_MAX_RETRIES = 10
JUDGE_MAX_WAIT_SECONDS = 60
RELEVANCY_QUESTIONS_PER_ANSWER = 1
JUDGE_PROBE = "Reply with the single word ok."


class EmbedderAdapter(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


def check_judge(judge_model: BaseChatModel) -> None:
    try:
        judge_model.invoke(JUDGE_PROBE)
    except Exception as error:
        raise JudgeUnavailable(
            f"judge model call failed before the run: {type(error).__name__}: {error}"
        ) from error


def judged_cases(results: Sequence[CaseResult]) -> list[CaseResult]:
    return [r for r in results if r.case.answerable and r.answer is not None and r.contexts]


def sample(result: CaseResult) -> SingleTurnSample:
    return SingleTurnSample(
        user_input=result.case.question,
        response=result.answer.text,
        retrieved_contexts=result.contexts,
        reference=result.case.reference,
    )


def score_value(value: object) -> float | None:
    if isinstance(value, int | float) and not math.isnan(value):
        return round(float(value), 4)
    return None


def judge(
    results: Sequence[CaseResult],
    judge_model: BaseChatModel,
    embedder: Embedder,
    max_workers: int,
) -> dict:
    cases = judged_cases(results)
    if not cases:
        return {}
    usage = UsageMetadataCallbackHandler()
    run_config = RunConfig(
        timeout=JUDGE_TIMEOUT_SECONDS,
        max_retries=JUDGE_MAX_RETRIES,
        max_wait=JUDGE_MAX_WAIT_SECONDS,
        max_workers=max_workers,
    )
    outcome = evaluate(
        dataset=EvaluationDataset(samples=[sample(r) for r in cases]),
        metrics=[
            Faithfulness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ResponseRelevancy(strictness=RELEVANCY_QUESTIONS_PER_ANSWER),
        ],
        llm=LangchainLLMWrapper(judge_model, run_config=run_config),
        embeddings=LangchainEmbeddingsWrapper(EmbedderAdapter(embedder)),
        run_config=run_config,
        callbacks=[usage],
        raise_exceptions=False,
        show_progress=False,
    )
    for result, scores in zip(cases, outcome.scores, strict=True):
        result.metrics = {name: score_value(scores.get(name)) for name in RAGAS_METRICS}
    return usage.usage_metadata
