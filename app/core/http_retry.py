import time
from collections.abc import Callable

import httpx

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def post_with_retry(
    client: httpx.Client,
    path: str,
    payload: dict,
    *,
    label: str,
    fail: Callable[[str], Exception],
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    for attempt in range(max_retries + 1):
        is_last = attempt == max_retries
        try:
            response = client.post(path, json=payload)
        except httpx.TransportError as error:
            if is_last:
                raise fail(f"{label} request failed: {type(error).__name__}") from error
        else:
            if response.status_code == httpx.codes.OK:
                return response
            retryable = response.status_code in RETRYABLE_STATUSES
            if is_last or not retryable:
                raise fail(f"{label} request failed with status {response.status_code}")
        sleep(backoff_seconds * 2**attempt)
    raise fail(f"{label} request failed")
