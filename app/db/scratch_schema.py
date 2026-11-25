import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import sql

from app.core.config import Settings
from app.db.migrator import Migrator, load_migrations
from app.db.pool import to_psycopg_dsn


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(to_psycopg_dsn(settings.database_url), autocommit=True)


@contextmanager
def scratch_schema(settings: Settings, prefix: str) -> Iterator[str]:
    schema = f"{prefix}_{uuid.uuid4().hex[:12]}"
    name = sql.Identifier(schema)
    try:
        with connect(settings) as conn:
            conn.execute(sql.SQL("create schema {}").format(name))
            conn.execute(sql.SQL("set search_path to {}").format(name))
            Migrator(conn, load_migrations(), settings.embedding_dimension).apply_pending()
        yield schema
    finally:
        with connect(settings) as conn:
            conn.execute(sql.SQL("drop schema if exists {} cascade").format(name))


def copy_table(settings: Settings, schema: str, table: str, source_schema: str = "public") -> None:
    with connect(settings) as conn:
        conn.execute(
            sql.SQL("insert into {}.{} select * from {}.{}").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(source_schema),
                sql.Identifier(table),
            )
        )
