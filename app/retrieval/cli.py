import argparse
from collections.abc import Sequence

from app.composition import build_container
from app.core.config import get_settings
from app.exceptions.errors import AppError
from app.exceptions.handlers import report_cli_error

PREVIEW_CHARACTERS = 110


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.retrieval.cli")
    parser.add_argument("query")
    query = parser.parse_args(argv).query

    settings = get_settings()
    container = build_container(settings)
    container.pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    try:
        chunks = container.retriever.retrieve(query)
    except AppError as error:
        return report_cli_error(error)
    finally:
        container.pool.close()

    if not chunks:
        print("no matching chunks")
    for position, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.content.split())[:PREVIEW_CHARACTERS]
        print(
            f"{position}. score {chunk.score:.4f} vector {chunk.vector_rank} "
            f"fts {chunk.fts_rank} | {chunk.source}#{chunk.chunk_index} | {preview}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
