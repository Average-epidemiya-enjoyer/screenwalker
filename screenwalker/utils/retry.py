"""Retry decorator with configurable exponential back-off.

Provides a ``@retry(...)`` decorator that wraps synchronous functions and
retries them on specified exceptions up to a maximum number of attempts,
with exponential back-off and optional jitter.

Example:
    >>> from screenwalker.utils.retry import retry
    >>> from screenwalker.core.errors import ElementNotFound
    >>>
    >>> @retry(max_attempts=3, exceptions=(ElementNotFound,), backoff_base=1.5)
    ... def find_element(query: str):
    ...     ...
"""

from __future__ import annotations

import functools
import time
import random
from typing import Callable, Type, TypeVar

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable)


def retry(
    max_attempts: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    backoff_base: float = 1.5,
    backoff_max: float = 30.0,
    jitter: bool = True,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[F], F]:
    """Decorator factory that retries a function on specified exceptions.

    Uses exponential back-off:  ``delay = min(backoff_base ** attempt, backoff_max)``

    Args:
        max_attempts: Total number of attempts (1 = no retries).
        exceptions: Tuple of exception types that trigger a retry.
        backoff_base: Exponential base for delay calculation.
        backoff_max: Maximum delay cap in seconds.
        jitter: Add uniform random jitter (0–1 s) to avoid thundering herd.
        on_retry: Optional callback called before each retry with
            ``(attempt_number, exception)``.

    Returns:
        Decorator that wraps a callable with retry logic.

    Raises:
        The last exception raised if all attempts are exhausted.

    Example:
        >>> @retry(max_attempts=5, exceptions=(TimeoutError,), backoff_base=2.0)
        ... def unstable_call() -> str:
        ...     ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break

                    delay = min(backoff_base ** (attempt - 1), backoff_max)
                    if jitter:
                        delay += random.uniform(0, 1)

                    logger.debug(
                        "Retrying",
                        func=func.__qualname__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=round(delay, 2),
                        error=str(exc),
                    )

                    if on_retry:
                        on_retry(attempt, exc)

                    time.sleep(delay)

            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_until(
    condition: Callable[[], bool],
    timeout: float = 15.0,
    poll_interval: float = 0.5,
    description: str = "condition",
) -> bool:
    """Poll *condition* until it returns True or *timeout* elapses.

    Args:
        condition: Zero-argument callable returning True when satisfied.
        timeout: Maximum total wait time in seconds.
        poll_interval: Seconds between polls.
        description: Human-readable description for log messages.

    Returns:
        True if condition was satisfied within timeout, False otherwise.

    Example:
        >>> ok = retry_until(lambda: os.path.exists("/tmp/ready"), timeout=10)
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_interval)
    logger.warning("retry_until timed out", description=description, timeout=timeout)
    return False
