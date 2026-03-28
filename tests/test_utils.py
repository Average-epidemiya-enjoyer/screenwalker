"""Tests for screenwalker.utils.retry and screenwalker.utils.config."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from screenwalker.utils.retry import retry, retry_until
from screenwalker.utils.config import AppConfig, load_config, merge_scenario_config, _deep_merge


# ---------------------------------------------------------------------------
# retry decorator
# ---------------------------------------------------------------------------


class TestRetryDecorator:
    def test_succeeds_on_first_attempt(self) -> None:
        calls = []

        @retry(max_attempts=3, jitter=False)
        def fn() -> str:
            calls.append(1)
            return "ok"

        result = fn()
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_matching_exception(self) -> None:
        calls = []

        @retry(max_attempts=3, exceptions=(ValueError,), backoff_base=0.001, jitter=False)
        def fn() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "done"

        with patch("time.sleep"):
            result = fn()
        assert result == "done"
        assert len(calls) == 3

    def test_raises_after_max_attempts(self) -> None:
        @retry(max_attempts=2, exceptions=(RuntimeError,), backoff_base=0.001, jitter=False)
        def fn() -> None:
            raise RuntimeError("always fails")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="always fails"):
                fn()

    def test_does_not_retry_non_matching_exception(self) -> None:
        calls = []

        @retry(max_attempts=3, exceptions=(ValueError,), backoff_base=0.001, jitter=False)
        def fn() -> None:
            calls.append(1)
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            fn()
        assert len(calls) == 1

    def test_on_retry_callback_invoked(self) -> None:
        callback = MagicMock()

        @retry(max_attempts=3, exceptions=(ValueError,), backoff_base=0.001, jitter=False, on_retry=callback)
        def fn() -> str:
            if callback.call_count < 2:
                raise ValueError("retry me")
            return "ok"

        with patch("time.sleep"):
            fn()
        assert callback.call_count == 2

    def test_backoff_max_caps_delay(self) -> None:
        sleep_calls = []

        original_sleep = __import__("time").sleep

        @retry(max_attempts=4, exceptions=(ValueError,), backoff_base=100.0, backoff_max=0.5, jitter=False)
        def fn() -> None:
            raise ValueError("retry")

        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with pytest.raises(ValueError):
                fn()

        assert all(d <= 0.5 for d in sleep_calls)

    def test_max_attempts_one_means_no_retry(self) -> None:
        calls = []

        @retry(max_attempts=1, exceptions=(ValueError,))
        def fn() -> None:
            calls.append(1)
            raise ValueError("immediate")

        with pytest.raises(ValueError):
            fn()
        assert len(calls) == 1

    def test_wraps_preserves_function_name(self) -> None:
        @retry(max_attempts=2)
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"


# ---------------------------------------------------------------------------
# retry_until
# ---------------------------------------------------------------------------


class TestRetryUntil:
    def test_returns_true_when_condition_met_immediately(self) -> None:
        assert retry_until(lambda: True, timeout=1.0, poll_interval=0.01)

    def test_returns_true_when_condition_met_after_delay(self) -> None:
        calls = [False, False, True]
        it = iter(calls)
        with patch("time.sleep"):
            result = retry_until(lambda: next(it, True), timeout=10.0, poll_interval=0.01)
        assert result is True

    def test_returns_false_on_timeout(self) -> None:
        with patch("time.sleep"):
            with patch("time.monotonic", side_effect=[0.0, 0.0, 999.0]):
                result = retry_until(lambda: False, timeout=1.0, poll_interval=0.01)
        assert result is False


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_merges_flat_dicts(self) -> None:
        result = _deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merges_nested_dicts(self) -> None:
        base = {"outer": {"x": 1, "y": 2}}
        override = {"outer": {"y": 99, "z": 3}}
        result = _deep_merge(base, override)
        assert result == {"outer": {"x": 1, "y": 99, "z": 3}}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}

    def test_override_replaces_non_dict_with_dict(self) -> None:
        result = _deep_merge({"key": "value"}, {"key": {"nested": True}})
        assert result == {"key": {"nested": True}}


# ---------------------------------------------------------------------------
# load_config / merge_scenario_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_returns_app_config_with_defaults(self) -> None:
        cfg = load_config(None)
        assert isinstance(cfg, AppConfig)

    def test_loads_user_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("timeouts:\n  step_default: 99\n", encoding="utf-8")
        cfg = load_config(yaml_file)
        assert cfg.timeouts.step_default == 99

    def test_raises_for_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestMergeScenarioConfig:
    def test_overrides_nested_value(self) -> None:
        base = load_config(None)
        overrides = {"timeouts": {"step_default": 123}}
        merged = merge_scenario_config(base, overrides)
        assert merged.timeouts.step_default == 123

    def test_does_not_mutate_base(self) -> None:
        base = load_config(None)
        original_timeout = base.timeouts.step_default
        merge_scenario_config(base, {"timeouts": {"step_default": 999}})
        assert base.timeouts.step_default == original_timeout
