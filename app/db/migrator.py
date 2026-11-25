import re
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.db.search_config import FULL_TEXT_LANGUAGE, MAX_INDEXED_DIMENSIONS
from app.exceptions.errors import MigrationError

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
ADVISORY_LOCK_ID = 7283410001
FILE_PATTERN = re.compile(
    r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.(?P<direction>up|down)\.sql$"
)


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    up_sql: str
    down_sql: str


def load_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    names: dict[str, str] = {}
    for path in sorted(directory.glob("*.sql")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            raise MigrationError(f"unrecognised migration file name: {path.name}")
        version = match["version"]
        grouped[version][match["direction"]] = path.read_text(encoding="utf-8")
        names[version] = match["name"]
    migrations = []
    for version in sorted(grouped):
        files = grouped[version]
        if set(files) != {"up", "down"}:
            raise MigrationError(f"migration {version} needs both up and down files")
        migrations.append(Migration(version, names[version], files["up"], files["down"]))
    return migrations


def render(sql: str, embedding_dimension: int) -> str:
    if not 1 <= embedding_dimension <= MAX_INDEXED_DIMENSIONS:
        raise MigrationError(
            f"embedding dimension must be between 1 and {MAX_INDEXED_DIMENSIONS}"
        )
    return sql.replace("{{embedding_dimension}}", str(embedding_dimension)).replace(
        "{{fts_language}}", FULL_TEXT_LANGUAGE
    )


class Migrator:
    def __init__(
        self,
        conn: psycopg.Connection,
        migrations: list[Migration],
        embedding_dimension: int,
    ) -> None:
        self._conn = conn
        self._migrations = migrations
        self._embedding_dimension = embedding_dimension

    def applied_versions(self) -> list[str]:
        with self._locked():
            return self._read_applied()

    def pending(self) -> list[Migration]:
        applied = set(self.applied_versions())
        return [m for m in self._migrations if m.version not in applied]

    def apply_pending(self) -> list[str]:
        applied_now = []
        with self._locked():
            applied = set(self._read_applied())
            for migration in self._migrations:
                if migration.version in applied:
                    continue
                self._run(migration.up_sql, migration.version, record=True)
                applied_now.append(migration.version)
        return applied_now

    def rollback_last(self) -> str | None:
        with self._locked():
            applied = self._read_applied()
            if not applied:
                return None
            version = applied[-1]
            migration = next((m for m in self._migrations if m.version == version), None)
            if migration is None:
                raise MigrationError(f"no migration files for applied version {version}")
            self._run(migration.down_sql, version, record=False)
            return version

    def verify_embedding_dimension(self) -> None:
        row = self._conn.execute(
            "select atttypmod from pg_attribute "
            "where attrelid = to_regclass('chunks') and attname = 'embedding'"
        ).fetchone()
        if row is None:
            raise MigrationError("chunks.embedding column does not exist")
        if row[0] != self._embedding_dimension:
            raise MigrationError(
                f"chunks.embedding has {row[0]} dimensions, settings expect "
                f"{self._embedding_dimension}"
            )

    def _run(self, sql: str, version: str, record: bool) -> None:
        with self._conn.transaction():
            self._conn.execute(render(sql, self._embedding_dimension))
            if record:
                self._conn.execute(
                    "insert into schema_migrations (version) values (%s)", (version,)
                )
            else:
                self._conn.execute(
                    "delete from schema_migrations where version = %s", (version,)
                )

    def _read_applied(self) -> list[str]:
        self._conn.execute(
            "create table if not exists schema_migrations ("
            "version text primary key, "
            "applied_at timestamptz not null default now())"
        )
        rows = self._conn.execute(
            "select version from schema_migrations order by version"
        ).fetchall()
        return [row[0] for row in rows]

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._conn.execute("select pg_advisory_lock(%s)", (ADVISORY_LOCK_ID,))
        try:
            yield
        finally:
            self._conn.execute("select pg_advisory_unlock(%s)", (ADVISORY_LOCK_ID,))
