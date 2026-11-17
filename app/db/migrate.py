import argparse
import math
import sys
from collections.abc import Sequence

import psycopg

from app.core.config import get_settings
from app.db.connection import to_psycopg_dsn
from app.db.migrator import MigrationError, Migrator, load_migrations


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
        except MigrationError as error:
            print(f"migration error: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
