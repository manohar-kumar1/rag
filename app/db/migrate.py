import argparse
import math
from collections.abc import Sequence

import psycopg

from app.core.config import get_settings
from app.db.migrator import Migrator, load_migrations
from app.db.pool import to_psycopg_dsn
from app.exceptions.errors import AppError
from app.exceptions.handlers import report_cli_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.db.migrate")
    parser.add_argument("command", nargs="?", choices=["up", "down", "status"], default="up")
    args = parser.parse_args(argv)

    settings = get_settings()
    with psycopg.connect(
        to_psycopg_dsn(settings.database_url),
        autocommit=True,
        connect_timeout=math.ceil(settings.db_pool_timeout_seconds),
    ) as conn:
        migrator = Migrator(conn, load_migrations(), settings.embedding_dimension)
        try:
            if args.command == "up":
                applied = migrator.apply_pending()
                migrator.verify_embedding_dimension()
                print(f"applied: {', '.join(applied) if applied else 'nothing to do'}")
            elif args.command == "down":
                version = migrator.rollback_last()
                print(f"rolled back: {version or 'nothing applied'}")
            else:
                print(f"applied: {', '.join(migrator.applied_versions()) or 'none'}")
                print(f"pending: {', '.join(m.version for m in migrator.pending()) or 'none'}")
        except AppError as error:
            return report_cli_error(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
