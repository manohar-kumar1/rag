import argparse
import json
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from app.composition import build_container
from app.core.config import Settings, get_settings
from app.db.connection import create_pool
from app.db.scratch import scratch_schema
from app.evaluation.__main__ import with_overrides
from app.evaluation.retrieval_scoring import RetrievalCase, score_case, summarize
from app.exceptions.exception import AppError
from app.exceptions.handlers import report_cli_error

SCHEMA_PREFIX = "retrieval_eval"
INDEXING_SETTINGS = (
    "chunk_size",
    "chunk_overlap",
    "embedding_model",
    "contextual_chunking_enabled",
    "contextual_model",
)


def load_cases(path: Path) -> list[RetrievalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RetrievalCase.model_validate_json(line) for line in lines if line.strip()]


def load_variants(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {label: [f"{name}={value}" for name, value in sets.items()] for label, sets in raw.items()}


def indexing_key(settings: Settings) -> tuple:
    return tuple(getattr(settings, name) for name in INDEXING_SETTINGS)


def run_variant(
    settings: Settings, schema: str, cases: Sequence[RetrievalCase]
) -> tuple[dict, float]:
    pool = create_pool(settings, search_path=schema)
    container = build_container(settings, pool)
    pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    try:
        started = time.perf_counter()
        scores = [score_case(case, container.retriever.retrieve(case.question)) for case in cases]
        elapsed = time.perf_counter() - started
    finally:
        pool.close()
    return summarize(scores), elapsed / len(cases)


def ingest(settings: Settings, schema: str, files: Sequence[Path]) -> dict:
    pool = create_pool(settings, search_path=schema)
    container = build_container(settings, pool)
    pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    try:
        started = time.perf_counter()
        for path in files:
            container.ingestion.ingest(path.read_bytes(), source=path.name, title=path.stem)
        elapsed = time.perf_counter() - started
        with pool.connection() as conn:
            chunks = conn.execute("select count(*) from chunks").fetchone()[0]
    finally:
        pool.close()
    return {"chunks": chunks, "seconds": round(elapsed, 1)}


def evaluate(
    base: Settings, files: Sequence[Path], cases: Sequence[RetrievalCase], variants: dict
) -> dict:
    report: dict = {"variants": {}}
    by_indexing: dict[tuple, tuple[str, dict]] = {}
    with ExitStack() as stack:
        for label, pairs in variants.items():
            settings = with_overrides(base, pairs)
            key = indexing_key(settings)
            if key not in by_indexing:
                schema = stack.enter_context(scratch_schema(settings, SCHEMA_PREFIX))
                by_indexing[key] = (schema, ingest(settings, schema, files))
            schema, indexing = by_indexing[key]
            summary, seconds = run_variant(settings, schema, cases)
            report["variants"][label] = {
                "settings": pairs,
                "indexing": indexing,
                "seconds_per_question": round(seconds, 3),
                **summary,
            }
            print(
                f"{label}: {json.dumps(summary['hit_at'])} "
                f"mrr {summary['mrr']} recall {summary['page_recall']}",
                flush=True,
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.retrieval_eval")
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not all(path.is_file() for path in [*args.files, args.cases, args.variants]):
        print("every input must be an existing file", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            get_settings(),
            args.files,
            load_cases(args.cases),
            load_variants(args.variants),
        )
    except (AppError, ValueError) as error:
        if isinstance(error, AppError):
            return report_cli_error(error)
        print(f"invalid input: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
