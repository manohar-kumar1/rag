from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool

from app.composition import Container, build_container
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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


@app.get("/health")
def health(pool: Annotated[ConnectionPool, Depends(get_pool)]) -> dict[str, str]:
    try:
        with pool.connection() as conn:
            conn.execute("select 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}
