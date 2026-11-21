from fastapi import APIRouter

from app.api.documents import router as documents_router
from app.api.jobs import router as jobs_router
from app.api.query import router as query_router

API_V1_PREFIX = "/v1"

v1_router = APIRouter(prefix=API_V1_PREFIX)
v1_router.include_router(documents_router)
v1_router.include_router(jobs_router)
v1_router.include_router(query_router)
