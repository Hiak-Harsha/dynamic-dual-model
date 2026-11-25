"""
retry_util.py — Transient API error retry with exponential backoff and jitter.

Catches HTTP 429, 500, 502, 503, 504, as well as SDK error messages
mentioning 'rate limit', 'overloaded', or 'timeout'.
"""

from __future__ import annotations
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_status: set = {429, 500, 502, 503, 504},
) -> T:
    """Exponential backoff with jitter for transient API errors."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            if "credit_balance_exhausted" in msg or "insufficient_quota" in msg:
                raise
            is_retryable = (
                status in retryable_status
                or "rate limit" in msg
                or "overloaded" in msg
                or "timeout" in msg
            )
            if not is_retryable or attempt == max_attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= 0.5 + random.random()  # jitter
            time.sleep(delay)
    raise last_exc
