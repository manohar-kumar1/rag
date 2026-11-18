from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from psycopg_pool import ConnectionPool

from app.composition import Container, build_container
from app.core.config import get_settings
from app.exceptions.exception import DatabaseUnavailable
from app.exceptions.handlers import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    container = build_container(settings)
    container.pool.open(wait=True, timeout=settings.db_pool_timeout_seconds)
    app.state.container = container
    yield
    container.pool.close()


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_pool(
    container: Annotated[Container, Depends(get_container)],
) -> ConnectionPool:
    return container.pool


app = FastAPI(title="RAG", lifespan=lifespan)
register_exception_handlers(app)


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"ping": "pong"}


@app.get("/health")
def health(pool: Annotated[ConnectionPool, Depends(get_pool)]) -> dict[str, str]:
    try:
        with pool.connection() as conn:
            conn.execute("select 1")
    except Exception as exc:
        raise DatabaseUnavailable from exc
    return {"status": "ok"}
