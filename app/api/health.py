from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_readiness
from app.api.error_responses import error_responses
from app.core.readiness import ReadinessChecker

router = APIRouter(tags=["system"])


@router.get("/healthz", summary="Liveness probe, checks nothing else")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/readyz",
    summary="Readiness probe, checks every critical dependency",
    responses=error_responses(503),
)
def readyz(
    readiness: Annotated[ReadinessChecker, Depends(get_readiness)],
) -> dict[str, str | dict[str, str]]:
    return {"status": "ready", "checks": readiness.check()}
