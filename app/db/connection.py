from psycopg_pool import ConnectionPool

from app.core.config import Settings


def to_psycopg_dsn(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def create_pool(settings: Settings) -> ConnectionPool:
    return ConnectionPool(
        to_psycopg_dsn(settings.database_url),
        min_size=1,
        max_size=5,
        open=False,
    )
