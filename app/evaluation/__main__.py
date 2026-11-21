import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

from app.composition import build_container
from app.core.config import Settings, get_settings
from app.db.connection import create_pool
from app.db.scratch import scratch_schema
from app.evaluation.dataset import load_cases
from app.evaluation.scoring import answer_cases, case_row, summarize
from app.exceptions.exception import AppError
from app.exceptions.handlers import report_cli_error
from app.ingestion.upload import EXTENSION_TYPES
from app.llm.factory import create_judge_model
from app.observability.tracing import flush

SCHEMA_PREFIX = "eval"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def corpus_files(corpus: Path) -> list[Path]:
    return sorted(p for p in corpus.iterdir() if p.suffix.lower() in EXTENSION_TYPES)


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def run_metadata(settings: Settings, dataset: Path, files: list[Path]) -> dict:
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": {"file": dataset.name, "sha256": digest(dataset)},
        "corpus": {p.name: digest(p) for p in files},
        "models": {
            "primary": settings.google_model,
            "fallback": settings.anthropic_model if settings.anthropic_fallback_enabled else None,
            "decision": settings.google_decision_model,
            "judge": settings.eval_judge_model,
            "embedding": settings.embedding_model,
        },
        "retrieval": {
            "rrf_k": settings.rrf_k,
            "vector_weight": settings.retrieval_vector_weight,
            "fts_weight": settings.retrieval_fts_weight,
            "vector_candidates": settings.retrieval_vector_candidates,
            "fts_candidates": settings.retrieval_fts_candidates,
            "top_k": settings.retrieval_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "features": {
            "rerank": settings.rerank_model if settings.rerank_enabled else None,
            "rerank_candidates": settings.rerank_candidates,
            "rerank_top_n": settings.rerank_top_n,
            "contextual_chunking": (
                settings.contextual_model if settings.contextual_chunking_enabled else None
            ),
            "groundedness_check": settings.groundedness_check_enabled,
            "answer_cache": settings.answer_cache_enabled,
            "agent_max_iterations": settings.agent_max_iterations,
        },
    }


def parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    overrides = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"expected NAME=VALUE, got {pair!r}")
        overrides[name.strip().lower()] = value
    return overrides


def with_overrides(settings: Settings, pairs: Sequence[str]) -> Settings:
    overrides = parse_overrides(pairs)
    unknown = set(overrides) - set(Settings.model_fields)
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
    return Settings.model_validate({**settings.model_dump(), **overrides})


def evaluate_corpus(
    settings: Settings, corpus: Path, dataset: Path, skip_judge: bool
) -> dict:
    cases = load_cases(dataset)
    files = corpus_files(corpus)
    report = {"run": run_metadata(settings, dataset, files)}
    judge_model = None
    if not skip_judge:
        from app.evaluation.judge import check_judge

        judge_model = create_judge_model(settings)
        check_judge(judge_model)
    started = time.perf_counter()
    with scratch_schema(settings, SCHEMA_PREFIX) as schema:
        pool = create_pool(settings, search_path=schema)
        container = build_container(settings, pool)
        pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
        try:
            with get_usage_metadata_callback() as ingestion_usage:
                for path in files:
                    container.ingestion.ingest(
                        path.read_bytes(), source=path.name, title=path.stem
                    )
            with get_usage_metadata_callback() as answering_usage:
                results = asyncio.run(answer_cases(container.answerer, cases))
            report["ingestion_usage"] = ingestion_usage.usage_metadata
            report["answering_usage"] = answering_usage.usage_metadata
            usage = {}
            if judge_model is not None:
                from app.evaluation.judge import judge

                usage = judge(
                    results, judge_model, container.embedder, settings.eval_max_workers
                )
        finally:
            flush(container.tracing)
            pool.close()
    report["run"]["duration_seconds"] = round(time.perf_counter() - started, 1)
    report["judge_usage"] = usage
    report["summary"] = summarize(results)
    report["cases"] = [case_row(r) for r in results]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    if not args.corpus.is_dir() or not args.dataset.is_file():
        print("corpus must be a folder and dataset a file", file=sys.stderr)
        return 2
    try:
        settings = with_overrides(get_settings(), args.set)
    except ValueError as error:
        print(f"invalid --set: {error}", file=sys.stderr)
        return 2
    if not args.skip_judge and not settings.evaluation_configured:
        print("set EVAL_JUDGE_MODEL and EVAL_MAX_WORKERS, or pass --skip-judge", file=sys.stderr)
        return 2
    if not args.trace:
        settings = settings.model_copy(update={"langsmith_tracing": False})
    try:
        report = evaluate_corpus(settings, args.corpus, args.dataset, args.skip_judge)
    except AppError as error:
        return report_cli_error(error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
