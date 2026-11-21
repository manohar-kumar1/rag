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
    }


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
            for path in files:
                container.ingestion.ingest(path.read_bytes(), source=path.name, title=path.stem)
            results = asyncio.run(answer_cases(container.answerer, cases))
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
    args = parser.parse_args(argv)
    if not args.corpus.is_dir() or not args.dataset.is_file():
        print("corpus must be a folder and dataset a file", file=sys.stderr)
        return 2
    settings = get_settings()
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
