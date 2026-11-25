from psycopg_pool import ConnectionPool

from app.core.config import Settings


def to_psycopg_dsn(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def create_pool(settings: Settings, search_path: str | None = None) -> ConnectionPool:
    return ConnectionPool(
        to_psycopg_dsn(settings.database_url),
        kwargs={"options": f"-c search_path={search_path}"} if search_path else None,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        max_lifetime=settings.db_pool_max_lifetime_seconds,
        timeout=settings.db_pool_timeout_seconds,
        check=ConnectionPool.check_connection,
        open=False,
    )
