import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    retry_statuses: frozenset[int] = RETRY_STATUSES,
) -> T:
    """Retry transport failures and selected HTTP statuses. 401, 403, and 404 are not retried."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except httpx.TransportError as exc:
            last_error = exc
            if attempt == attempts:
                raise
            sleep(min(2 ** (attempt - 1), 8))
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status not in retry_statuses or attempt == attempts:
                raise
            retry_after = exc.response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = min(2 ** (attempt - 1), 8)
            sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop ended without a result")
