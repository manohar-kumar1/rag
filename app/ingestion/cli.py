import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.composition import build_container
from app.core.config import get_settings
from app.exceptions.errors import AppError
from app.exceptions.handlers import report_cli_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.cli")
    parser.add_argument("path", type=Path)
    path = parser.parse_args(argv).path

    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2

    settings = get_settings()
    container = build_container(settings)
    container.pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    try:
        result = container.ingestion.ingest(
            path.read_bytes(), source=path.name, title=path.stem
        )
    except AppError as error:
        return report_cli_error(error)
    finally:
        container.pool.close()

    outcome = "created" if result.created else "already ingested"
    print(f"document {result.document_id} {outcome}, {result.chunk_count} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
