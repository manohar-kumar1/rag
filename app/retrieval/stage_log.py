import json
import logging
from typing import Any

logger = logging.getLogger("rag.retrieval")


def log_stage(stage: str, **fields: Any) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps({"stage": stage, **fields}, default=str, ensure_ascii=False))
