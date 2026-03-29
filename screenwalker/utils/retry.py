"""Утилиты повторных попыток: устаревший декоратор и структурированный RetryStrategy.

Example (устаревший вариант)::

    >>> from screenwalker.utils.retry import retry
    >>> @retry(max_attempts=3, exceptions=(TimeoutError,), backoff_base=1.5)
    ... def unstable(): ...

Example (структурированный вариант)::

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
    """Структурированная конфигурация повторных попыток.

    Attributes:
        max_retries: Общее число попыток (1 = без повторов).
        backoff: Стратегия увеличения задержки — ``"linear"`` или ``"exponential"``.
        base_delay: Начальная задержка в секундах.
        max_delay: Максимальная задержка в секундах.
        jitter: Добавить до 1 с равномерного случайного джиттера.
    """

    max_retries: int = 3
    backoff: Literal["linear", "exponential"] = "exponential"
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = False

    def delay_for(self, attempt: int) -> float:
        """Вычисляет время ожидания перед указанной попыткой (нумерация с 1).

        Для первой попытки возвращает 0 (задержка перед первой попыткой не нужна).

        Args:
            attempt: Номер попытки (нумерация с 1).

        Returns:
            Задержка в секундах, ограниченная :attr:`max_delay`.
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
    """Декоратор, повторяющий вызов функции согласно *strategy*.

    Args:
        strategy: :class:`RetryStrategy`, управляющий задержками и числом попыток.
        exceptions: Типы исключений, вызывающие повтор.
        on_retry: Необязательный колбэк ``(attempt, exc)``, вызываемый перед каждым повтором.

    Returns:
        Декоратор, оборачивающий функцию логикой повторных попыток.

    Raises:
        Последнее возникшее исключение после исчерпания всех попыток.
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
# Устаревший декоратор retry
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    backoff_base: float = 1.5,
    backoff_max: float = 30.0,
    jitter: bool = True,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[F], F]:
    """Фабрика декораторов, повторяющих функцию при указанных исключениях.

    Использует экспоненциальное увеличение задержки: ``delay = min(backoff_base ** attempt, backoff_max)``

    Args:
        max_attempts: Общее число попыток (1 = без повторов).
        exceptions: Кортеж типов исключений, вызывающих повтор.
        backoff_base: Основание степени для вычисления задержки.
        backoff_max: Максимальная задержка в секундах.
        jitter: Добавить равномерный случайный джиттер (0–1 с).
        on_retry: Необязательный колбэк ``(номер_попытки, исключение)`` перед повтором.

    Returns:
        Декоратор, оборачивающий функцию логикой повторных попыток.

    Raises:
        Последнее возникшее исключение после исчерпания всех попыток.
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
    """Опрашивает *condition* до возврата True или истечения *timeout*.

    Args:
        condition: Вызываемый без аргументов объект, возвращающий True при выполнении условия.
        timeout: Максимальное общее время ожидания в секундах.
        poll_interval: Интервал между опросами в секундах.
        description: Читаемое описание для сообщений лога.

    Returns:
        True если условие выполнено в рамках таймаута, иначе False.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_interval)
    logger.warning("retry_until timed out", description=description, timeout=timeout)
    return False
