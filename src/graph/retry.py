"""Per-tool retry, timeout and backoff.

Timeouts use a worker thread rather than signals so this works inside the API
server and Streamlit, both of which run the graph off the main thread.
A timed-out call is abandoned, not killed — so tools must be idempotent or
gated, which is why every irreversible tool here sits behind human approval.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

RETRYABLE = (ConnectionError, TimeoutError, FuturesTimeout, OSError)


class ToolTimeout(Exception):
    pass


@dataclass
class Attempted:
    ok: bool
    value: Any = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0


def call_with_retry(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    retries: int = 2,
    timeout_s: float = 10.0,
    backoff_base_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> Attempted:
    """Run `fn(**kwargs)`, retrying transient failures with exponential backoff.

    Non-transient errors (a TypeError from bad arguments, a ValueError from
    validation) fail immediately: retrying a deterministic bug just wastes the
    budget.
    """
    started = time.perf_counter()
    last_error: str | None = None

    for attempt in range(1, retries + 2):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fn, **kwargs)
                value = future.result(timeout=timeout_s)
            return Attempted(
                ok=True,
                value=value,
                attempts=attempt,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except FuturesTimeout:
            last_error = f"ToolTimeout: exceeded {timeout_s}s"
        except RETRYABLE as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            return Attempted(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                attempts=attempt,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if attempt <= retries:
            sleep(backoff_base_s * (2 ** (attempt - 1)))

    return Attempted(
        ok=False,
        error=last_error,
        attempts=retries + 1,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
