from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool

from app.core.config import get_settings
from app.db.connection import create_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = create_pool(get_settings())
    pool.open(wait=True, timeout=15)
    app.state.pool = pool
    yield
    pool.close()


def get_pool(request: Request) -> ConnectionPool:
    return request.app.state.pool


app = FastAPI(title="RAG", lifespan=lifespan)


@app.get("/health")
def health(pool: Annotated[ConnectionPool, Depends(get_pool)]) -> dict[str, str]:
    try:
        with pool.connection() as conn:
            conn.execute("select 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}
