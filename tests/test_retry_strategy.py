"""Tests for RetryStrategy and with_retry decorator."""

from __future__ import annotations

import pytest

from screenwalker.utils.retry import RetryStrategy, with_retry


class TestRetryStrategyDelays:
    def test_first_attempt_has_no_delay(self) -> None:
        s = RetryStrategy(base_delay=2.0, backoff="exponential")
        assert s.delay_for(1) == 0.0

    def test_zero_attempt_has_no_delay(self) -> None:
        s = RetryStrategy(base_delay=2.0, backoff="exponential")
        assert s.delay_for(0) == 0.0

    def test_exponential_second_attempt(self) -> None:
        s = RetryStrategy(base_delay=1.0, backoff="exponential", jitter=False)
        assert s.delay_for(2) == 1.0

    def test_exponential_third_attempt(self) -> None:
        s = RetryStrategy(base_delay=1.0, backoff="exponential", jitter=False)
        assert s.delay_for(3) == 2.0

    def test_exponential_fourth_attempt(self) -> None:
        s = RetryStrategy(base_delay=1.0, backoff="exponential", jitter=False)
        assert s.delay_for(4) == 4.0

    def test_linear_second_attempt(self) -> None:
        s = RetryStrategy(base_delay=2.0, backoff="linear", jitter=False)
        assert s.delay_for(2) == 2.0

    def test_linear_third_attempt(self) -> None:
        s = RetryStrategy(base_delay=2.0, backoff="linear", jitter=False)
        assert s.delay_for(3) == 4.0

    def test_max_delay_caps_exponential(self) -> None:
        s = RetryStrategy(base_delay=1.0, backoff="exponential", max_delay=3.0, jitter=False)
        assert s.delay_for(10) == 3.0

    def test_max_delay_caps_linear(self) -> None:
        s = RetryStrategy(base_delay=5.0, backoff="linear", max_delay=7.0, jitter=False)
        assert s.delay_for(5) == 7.0

    def test_jitter_adds_positive_value(self) -> None:
        s = RetryStrategy(base_delay=1.0, backoff="exponential", jitter=True)
        delay = s.delay_for(2)
        assert delay >= 1.0
        assert delay <= 2.0 + 1e-9  # base + max jitter (1.0)


class TestWithRetry:
    def test_succeeds_on_first_attempt(self) -> None:
        strategy = RetryStrategy(max_retries=3, base_delay=0.0)
        calls = []

        @with_retry(strategy)
        def fn():
            calls.append(1)
            return "ok"

        result = fn()
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_matching_exception(self) -> None:
        strategy = RetryStrategy(max_retries=3, base_delay=0.0)
        calls = []

        @with_retry(strategy, exceptions=(ValueError,))
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "done"

        result = fn()
        assert result == "done"
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self) -> None:
        strategy = RetryStrategy(max_retries=2, base_delay=0.0)

        @with_retry(strategy, exceptions=(RuntimeError,))
        def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            fn()

    def test_does_not_retry_non_matching_exception(self) -> None:
        strategy = RetryStrategy(max_retries=3, base_delay=0.0)
        calls = []

        @with_retry(strategy, exceptions=(ValueError,))
        def fn():
            calls.append(1)
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            fn()
        assert len(calls) == 1

    def test_on_retry_callback_called(self) -> None:
        strategy = RetryStrategy(max_retries=3, base_delay=0.0)
        retry_log: list[tuple[int, Exception]] = []

        @with_retry(strategy, on_retry=lambda a, e: retry_log.append((a, e)))
        def fn():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fn()

        assert len(retry_log) == 2  # called before attempt 2 and 3
        assert retry_log[0][0] == 1

    def test_preserves_function_name(self) -> None:
        strategy = RetryStrategy(max_retries=1, base_delay=0.0)

        @with_retry(strategy)
        def my_function():
            return 42

        assert my_function.__name__ == "my_function"

    def test_max_retries_one_means_no_retry(self) -> None:
        strategy = RetryStrategy(max_retries=1, base_delay=0.0)
        calls = []

        @with_retry(strategy)
        def fn():
            calls.append(1)
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            fn()
        assert len(calls) == 1
