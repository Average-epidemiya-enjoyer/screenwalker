"""Retry utilities: legacy decorator and structured RetryStrategy.

Example (legacy)::

    >>> from screenwalker.utils.retry import retry
    >>> @retry(max_attempts=3, exceptions=(TimeoutError,), backoff_base=1.5)
    ... def unstable(): ...

Example (structured)::

    >>> from screenwalker.utils.retry import RetryStrategy, with_retry
    >>> strategy = RetryStrategy(max_retries=3, backoff="exponential", base_delay=0.5)
    >>> @with_retry(strategy, exceptions=(TimeoutError,))
    ... def unstable(): ...
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import Callable, Literal, Type, TypeVar

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable)


# ---------------------------------------------------------------------------
# RetryStrategy
# ---------------------------------------------------------------------------


@dataclass
class RetryStrategy:
    """Structured retry configuration.

    Attributes:
        max_retries: Total number of attempts (1 = no retries).
        backoff: Delay growth strategy — ``"linear"`` or ``"exponential"``.
        base_delay: Starting delay in seconds.
        max_delay: Upper cap on delay in seconds.
        jitter: Add up to 1 s of uniform random jitter.
    """

    max_retries: int = 3
    backoff: Literal["linear", "exponential"] = "exponential"
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = False

    def delay_for(self, attempt: int) -> float:
        """Compute wait duration before the given 1-based attempt.

        Returns 0 for the first attempt (no delay before the first try).

        Args:
            attempt: 1-based attempt number.

        Returns:
            Delay in seconds, capped at :attr:`max_delay`.
        """
        if attempt <= 1:
            return 0.0
        step = attempt - 1
        if self.backoff == "linear":
            d = self.base_delay * step
        else:
            d = self.base_delay * (2.0 ** (step - 1))
        d = min(d, self.max_delay)
        if self.jitter:
            d += random.uniform(0.0, 1.0)
        return d


def with_retry(
    strategy: RetryStrategy,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[F], F]:
    """Decorator that retries a function according to *strategy*.

    Args:
        strategy: :class:`RetryStrategy` controlling delays and attempt count.
        exceptions: Exception types that trigger a retry.
        on_retry: Optional callback ``(attempt, exc)`` called before each retry.

    Returns:
        Decorator wrapping a callable with retry logic.

    Raises:
        The last exception if all attempts are exhausted.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, strategy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == strategy.max_retries:
                        break
                    delay = strategy.delay_for(attempt + 1)
                    logger.debug(
                        "with_retry.retrying",
                        func=func.__qualname__,
                        attempt=attempt,
                        max_retries=strategy.max_retries,
                        delay=round(delay, 3),
                        error=str(exc),
                    )
                    if on_retry:
                        on_retry(attempt, exc)
                    if delay > 0:
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Legacy retry decorator
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    backoff_base: float = 1.5,
    backoff_max: float = 30.0,
    jitter: bool = True,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[F], F]:
    """Decorator factory that retries a function on specified exceptions.

    Uses exponential back-off: ``delay = min(backoff_base ** attempt, backoff_max)``

    Args:
        max_attempts: Total number of attempts (1 = no retries).
        exceptions: Tuple of exception types that trigger a retry.
        backoff_base: Exponential base for delay calculation.
        backoff_max: Maximum delay cap in seconds.
        jitter: Add uniform random jitter (0–1 s).
        on_retry: Optional callback ``(attempt_number, exception)`` before each retry.

    Returns:
        Decorator that wraps a callable with retry logic.

    Raises:
        The last exception raised if all attempts are exhausted.
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
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_interval)
    logger.warning("retry_until timed out", description=description, timeout=timeout)
    return False
