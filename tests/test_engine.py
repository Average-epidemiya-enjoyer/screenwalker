"""Tests for ScenarioEngine, RunContext, and the step-execution state machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from screenwalker.core.context import RunContext, StepRecord
from screenwalker.core.engine import ScenarioEngine, _build_region
from screenwalker.core.errors import (
    ActionError,
    ElementNotFound,
    ScenarioError,
    ScenarioValidationError,
    ScreenMismatch,
    StepTimeout,
)
from screenwalker.core.step import FindMethod, FindSpec, OnFailure, Step, StepAction
from screenwalker.learning.cache import ActionCache, CacheEntry
from screenwalker.utils.config import AppConfig
from screenwalker.vision.screen_state import BBox, FindResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image() -> Image.Image:
    return Image.new("RGB", (1920, 1080), color=(200, 200, 200))


def _find_result(
    element: str = "OK",
    x: int = 100,
    y: int = 200,
    w: int = 80,
    h: int = 30,
    confidence: float = 0.95,
    method: str = "ocr",
) -> FindResult:
    return FindResult(element=element, confidence=confidence, bbox=BBox(x, y, w, h), method=method)


def _make_engine(
    steps: list[Step],
    teardown: list[Step] | None = None,
    dry_run: bool = False,
    capture: Any = None,
    ocr_engine: Any = None,
    template_matcher: Any = None,
    mouse: Any = None,
    keyboard: Any = None,
    clipboard: Any = None,
    cache: Any = None,
    config: AppConfig | None = None,
) -> ScenarioEngine:
    """Construct an engine with all vision/action subsystems mocked by default."""
    cfg = config or AppConfig()
    cfg.timeouts.step_default = 2.0
    cfg.timeouts.poll_interval = 0.05
    cfg.timeouts.screenshot_delay = 0.0
    cfg.retry.backoff_base = 0.0
    cfg.retry.backoff_max = 0.0

    ctx = RunContext("test", dry_run=dry_run)

    if capture is None:
        capture = MagicMock()
        capture.capture_full.return_value = _make_image()

    if mouse is None:
        mouse = MagicMock()
    if keyboard is None:
        keyboard = MagicMock()
    if clipboard is None:
        clipboard = MagicMock()
        clipboard.copy_selected.return_value = "copied text"
    if cache is None:
        cache = MagicMock()
        cache.lookup.return_value = None  # miss by default

    step_logger = MagicMock()

    return ScenarioEngine(
        scenario_name="test",
        steps=steps,
        teardown_steps=teardown or [],
        context=ctx,
        config=cfg,
        capture=capture,
        ocr_engine=ocr_engine,
        template_matcher=template_matcher,
        mouse=mouse,
        keyboard=keyboard,
        clipboard=clipboard,
        cache=cache,
        step_logger=step_logger,
    )


# ---------------------------------------------------------------------------
# _build_region helper
# ---------------------------------------------------------------------------


class TestBuildRegion:
    def test_none_returns_none(self) -> None:
        assert _build_region(None) is None

    def test_string_returns_none(self) -> None:
        assert _build_region("title_bar") is None

    def test_list_returns_bbox(self) -> None:
        bbox = _build_region([10, 20, 300, 100])
        assert bbox == BBox(10, 20, 300, 100)

    def test_wrong_length_returns_none(self) -> None:
        assert _build_region([1, 2, 3]) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RunContext tests
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_interpolate_known_variable(self) -> None:
        ctx = RunContext("test", variables={"user": "admin"})
        assert ctx.interpolate("Hello {{ user }}!") == "Hello admin!"

    def test_interpolate_unknown_variable_unchanged(self) -> None:
        ctx = RunContext("test", variables={})
        assert "{{ unknown }}" in ctx.interpolate("Hello {{ unknown }}!")

    def test_interpolate_multiple_variables(self) -> None:
        ctx = RunContext("test", variables={"a": "foo", "b": "bar"})
        assert ctx.interpolate("{{ a }}:{{ b }}") == "foo:bar"

    def test_set_variable(self) -> None:
        ctx = RunContext("test")
        ctx.set_variable("x", "42")
        assert ctx.variables["x"] == "42"

    def test_record_step_increments_count(self) -> None:
        ctx = RunContext("test")
        ctx.record_step(StepRecord(step_id="s1", action="click", success=True, elapsed=0.1))
        assert ctx.step_count == 1

    def test_failed_steps_filtered(self) -> None:
        ctx = RunContext("test")
        ctx.record_step(StepRecord(step_id="ok", action="click", success=True, elapsed=0.1))
        ctx.record_step(
            StepRecord(step_id="fail", action="type", success=False, elapsed=0.2, error="oops")
        )
        assert len(ctx.failed_steps) == 1
        assert ctx.failed_steps[0].step_id == "fail"

    def test_save_screenshot(self, tmp_path: Path) -> None:
        ctx = RunContext("test", output_dir=tmp_path)
        img = _make_image()
        path = ctx.save_screenshot(img, "my_step")
        assert path.exists()
        assert path.suffix == ".png"

    def test_repr_contains_scenario_name(self) -> None:
        ctx = RunContext("my_scenario")
        assert "my_scenario" in repr(ctx)


# ---------------------------------------------------------------------------
# Step model tests
# ---------------------------------------------------------------------------


class TestStep:
    def test_valid_minimal_step(self) -> None:
        step = Step(id="s1", action=StepAction.WAIT, wait=1.0)
        assert step.id == "s1"
        assert step.retries == 3

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(Exception):
            Step(id="s1", action="fly")  # type: ignore[arg-type]

    def test_find_spec_threshold_bounds(self) -> None:
        with pytest.raises(Exception):
            FindSpec(method=FindMethod.OCR, query="ok", threshold=1.5)

    def test_find_spec_offset_must_be_two_ints(self) -> None:
        with pytest.raises(Exception):
            FindSpec(method=FindMethod.OCR, query="ok", offset=[10])

    def test_on_failure_defaults_to_abort(self) -> None:
        step = Step(id="s1", action=StepAction.WAIT)
        assert step.on_failure == OnFailure.ABORT

    def test_on_failure_retry(self) -> None:
        step = Step(id="s1", action=StepAction.WAIT, on_failure=OnFailure.RETRY)
        assert step.on_failure == OnFailure.RETRY

    def test_expect_screen_field(self) -> None:
        step = Step(id="s1", action=StepAction.WAIT, expect_screen="login")
        assert step.expect_screen == "login"

    def test_retries_non_negative(self) -> None:
        with pytest.raises(Exception):
            Step(id="s1", action=StepAction.WAIT, retries=-1)


# ---------------------------------------------------------------------------
# ScenarioEngine — factory and validation
# ---------------------------------------------------------------------------


class TestScenarioEngineFactory:
    def test_from_yaml_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ScenarioEngine.from_yaml("/nonexistent/scenario.yaml")

    def test_from_yaml_loads_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text(
            "name: Test\nvariables:\n  x: '1'\nsteps:\n  - id: s1\n    action: wait\n    wait: 0\n"
        )
        engine = ScenarioEngine.from_yaml(f, config=AppConfig())
        assert engine.scenario_name == "Test"
        assert len(engine.steps) == 1
        assert engine.context.variables["x"] == "1"

    def test_variable_overrides_take_precedence(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("name: Vars\nvariables:\n  env: prod\nsteps: []\n")
        engine = ScenarioEngine.from_yaml(
            f, config=AppConfig(), variable_overrides={"env": "staging"}
        )
        assert engine.context.variables["env"] == "staging"

    def test_parse_steps_invalid_action_raises(self) -> None:
        with pytest.raises(ScenarioValidationError):
            ScenarioEngine._parse_steps([{"id": "bad", "action": "fly_to_moon"}])

    def test_parse_steps_non_dict_raises(self) -> None:
        with pytest.raises(ScenarioValidationError):
            ScenarioEngine._parse_steps(["not a dict"])  # type: ignore[list-item]

    def test_from_yaml_non_mapping_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("- just a list\n")
        with pytest.raises(ScenarioValidationError):
            ScenarioEngine.from_yaml(f, config=AppConfig())


# ---------------------------------------------------------------------------
# dry_run output
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_prints_all_steps(self, capsys: pytest.CaptureFixture) -> None:
        steps = [
            Step(id="step_one", action=StepAction.WAIT, wait=0.0),
            Step(id="step_two", action=StepAction.SCREENSHOT),
        ]
        engine = _make_engine(steps)
        engine.dry_run()
        out = capsys.readouterr().out
        assert "step_one" in out
        assert "step_two" in out


# ---------------------------------------------------------------------------
# Action handlers — dry-run mode (no actual I/O)
# ---------------------------------------------------------------------------


class TestActionsViaDryRun:
    def _run_step(self, step: Step) -> RunContext:
        engine = _make_engine([step], dry_run=True)
        engine.run()
        return engine.context

    def test_wait_action_completes(self) -> None:
        ctx = self._run_step(Step(id="w", action=StepAction.WAIT, wait=0.0))
        assert ctx.step_count == 1
        assert ctx.failed_steps == []

    def test_screenshot_action_dry_run_skips_capture(self) -> None:
        # In dry_run mode dispatch is skipped but step still records
        ctx = self._run_step(Step(id="sc", action=StepAction.SCREENSHOT))
        assert ctx.step_count == 1

    def test_hotkey_dry_run_records(self) -> None:
        ctx = self._run_step(Step(id="hk", action=StepAction.HOTKEY, keys=["ctrl", "s"]))
        assert ctx.failed_steps == []


# ---------------------------------------------------------------------------
# Action handlers — live mock (non-dry-run)
# ---------------------------------------------------------------------------


class TestActionsLive:
    def test_wait_action_calls_sleep(self) -> None:
        step = Step(id="w", action=StepAction.WAIT, wait=0.05)
        engine = _make_engine([step])
        with patch("screenwalker.core.engine.time") as mock_time:
            mock_time.monotonic.side_effect = [0, 0, 1, 999]
            mock_time.sleep = MagicMock()
            engine._action_wait(step, None)
        mock_time.sleep.assert_called_once_with(0.05)

    def test_hotkey_calls_keyboard(self) -> None:
        step = Step(id="hk", action=StepAction.HOTKEY, keys=["ctrl", "s"])
        kb = MagicMock()
        engine = _make_engine([step], keyboard=kb)
        engine._action_hotkey(step, None)
        kb.hotkey.assert_called_once_with("ctrl", "s")

    def test_hotkey_without_keys_raises(self) -> None:
        step = Step(id="hk", action=StepAction.HOTKEY)
        engine = _make_engine([step])
        with pytest.raises(ActionError):
            engine._action_hotkey(step, None)

    def test_click_calls_mouse(self) -> None:
        step = Step(
            id="cl",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="OK"),
        )
        mouse = MagicMock()
        engine = _make_engine([step], mouse=mouse)
        result = _find_result(x=100, y=200, w=80, h=30)
        engine._action_click(step, result)
        mouse.click.assert_called_once_with(140, 215, button="left")  # center of bbox

    def test_click_without_result_raises(self) -> None:
        step = Step(
            id="cl",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="OK"),
        )
        engine = _make_engine([step])
        with pytest.raises(ElementNotFound):
            engine._action_click(step, None)

    def test_click_applies_offset(self) -> None:
        step = Step(
            id="cl",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="Label", offset=[20, -5]),
        )
        mouse = MagicMock()
        engine = _make_engine([step], mouse=mouse)
        result = _find_result(x=100, y=200, w=80, h=30)  # center = 140, 215
        engine._action_click(step, result)
        mouse.click.assert_called_once_with(160, 210, button="left")

    def test_double_click_calls_mouse(self) -> None:
        step = Step(
            id="dc",
            action=StepAction.DOUBLE_CLICK,
            find=FindSpec(method=FindMethod.OCR, query="OK"),
        )
        mouse = MagicMock()
        engine = _make_engine([step], mouse=mouse)
        result = _find_result()
        engine._action_double_click(step, result)
        mouse.double_click.assert_called_once()

    def test_right_click_calls_mouse(self) -> None:
        step = Step(
            id="rc",
            action=StepAction.RIGHT_CLICK,
            find=FindSpec(method=FindMethod.OCR, query="item"),
        )
        mouse = MagicMock()
        engine = _make_engine([step], mouse=mouse)
        engine._action_right_click(step, _find_result())
        mouse.right_click.assert_called_once()

    def test_type_with_find_result_clicks_first(self) -> None:
        step = Step(
            id="ty",
            action=StepAction.TYPE,
            text="hello",
            find=FindSpec(method=FindMethod.OCR, query="input"),
        )
        mouse = MagicMock()
        kb = MagicMock()
        engine = _make_engine([step], mouse=mouse, keyboard=kb)
        engine._action_type(step, _find_result())
        mouse.click.assert_called_once()
        kb.type_text.assert_called_once_with("hello")

    def test_type_without_result_still_types(self) -> None:
        step = Step(id="ty", action=StepAction.TYPE, text="world")
        kb = MagicMock()
        engine = _make_engine([step], keyboard=kb)
        engine._action_type(step, None)
        kb.type_text.assert_called_once_with("world")

    def test_assert_visible_ok(self) -> None:
        step = Step(
            id="av",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="Submit"),
        )
        engine = _make_engine([step])
        engine._action_assert_visible(step, _find_result())  # should not raise

    def test_assert_visible_fails_without_result(self) -> None:
        step = Step(
            id="av",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="Submit"),
        )
        engine = _make_engine([step])
        with pytest.raises(ElementNotFound):
            engine._action_assert_visible(step, None)

    def test_assert_text_ok(self) -> None:
        step = Step(
            id="at",
            action=StepAction.ASSERT_TEXT,
            text="Hello",
            find=FindSpec(method=FindMethod.OCR, query="Hello", fuzzy_threshold=80),
        )
        engine = _make_engine([step])
        engine._action_assert_text(step, _find_result(element="Hello World!"))

    def test_assert_text_fails_on_mismatch(self) -> None:
        step = Step(
            id="at",
            action=StepAction.ASSERT_TEXT,
            text="Expected",
            find=FindSpec(method=FindMethod.OCR, query="Expected", fuzzy_threshold=90),
        )
        engine = _make_engine([step])
        with pytest.raises(ActionError):
            engine._action_assert_text(step, _find_result(element="something completely different"))

    def test_copy_reads_clipboard(self) -> None:
        step = Step(
            id="cp",
            action=StepAction.COPY,
            find=FindSpec(method=FindMethod.OCR, query="field"),
        )
        clip = MagicMock()
        clip.copy_selected.return_value = "clipboard_text"
        engine = _make_engine([step], clipboard=clip)
        engine._action_copy(step, _find_result())
        assert engine.context.variables.get("clipboard") == "clipboard_text"

    def test_screenshot_action_saves_file(self, tmp_path: Path) -> None:
        step = Step(id="sc", action=StepAction.SCREENSHOT, label="final")
        capture = MagicMock()
        capture.capture_full.return_value = _make_image()
        cfg = AppConfig()
        ctx = RunContext("test", output_dir=tmp_path)
        engine = ScenarioEngine(
            scenario_name="test",
            steps=[step],
            teardown_steps=[],
            context=ctx,
            config=cfg,
            capture=capture,
            mouse=MagicMock(),
            keyboard=MagicMock(),
            clipboard=MagicMock(),
            cache=MagicMock(lookup=MagicMock(return_value=None)),
            step_logger=MagicMock(),
        )
        engine._action_screenshot(step, None)
        screenshots = list(tmp_path.rglob("*.png"))
        assert len(screenshots) == 1


# ---------------------------------------------------------------------------
# Vision pipeline — _locate_with_retry
# ---------------------------------------------------------------------------


class TestLocateWithRetry:
    def test_ocr_find_returns_result(self) -> None:
        step = Step(
            id="find",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="OK", threshold=0.8),
            timeout=2.0,
            retries=1,
        )
        ocr = MagicMock()
        expected = _find_result(element="OK")
        ocr.find_text.return_value = expected
        engine = _make_engine([step], ocr_engine=ocr)

        result = engine._locate_with_retry(step)
        assert result is expected

    def test_template_find_converts_match_result(self) -> None:
        step = Step(
            id="find",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.TEMPLATE, query="ok_button", threshold=0.8),
            timeout=2.0,
            retries=1,
        )
        from screenwalker.vision.template_match import MatchResult
        match = MatchResult(
            template_name="ok_button",
            confidence=0.92,
            center=(140, 215),
            bbox=BBox(100, 200, 80, 30),
        )
        matcher = MagicMock()
        matcher.find_one.return_value = match
        engine = _make_engine([step], template_matcher=matcher)

        result = engine._locate_with_retry(step)
        assert result.method == "template"
        assert result.confidence == 0.92

    def test_fallback_used_when_primary_fails(self) -> None:
        step = Step(
            id="find",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.TEMPLATE, query="btn", threshold=0.8),
            fallback=FindSpec(method=FindMethod.OCR, query="OK", threshold=0.7),
            timeout=2.0,
            retries=1,
        )
        # Template match fails, OCR succeeds
        matcher = MagicMock()
        matcher.find_one.return_value = None
        ocr = MagicMock()
        ocr.find_text.return_value = _find_result(element="OK", method="ocr")
        engine = _make_engine([step], template_matcher=matcher, ocr_engine=ocr)

        result = engine._locate_with_retry(step)
        assert result.method == "ocr"

    def test_cache_hit_skips_vision(self) -> None:
        step = Step(
            id="find",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="Save"),
            timeout=2.0,
            retries=1,
        )
        cached = _find_result(element="Save", method="cache:ocr")
        cache = MagicMock()
        cache.lookup.return_value = cached
        ocr = MagicMock()
        engine = _make_engine([step], cache=cache, ocr_engine=ocr)

        result = engine._locate_with_retry(step)
        assert result is cached
        ocr.find_text.assert_not_called()

    def test_timeout_raises_step_timeout(self) -> None:
        step = Step(
            id="find",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="Missing"),
            timeout=0.1,
            retries=1,
        )
        ocr = MagicMock()
        ocr.find_text.return_value = None
        engine = _make_engine([step], ocr_engine=ocr)

        with pytest.raises(StepTimeout) as exc_info:
            engine._locate_with_retry(step)
        assert exc_info.value.step_id == "find"
        assert "Missing" in exc_info.value.query

    def test_no_finder_configured_raises_timeout(self) -> None:
        step = Step(
            id="find",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="x"),
            timeout=0.1,
            retries=1,
        )
        # No OCR engine
        engine = _make_engine([step], ocr_engine=None)
        with pytest.raises(StepTimeout):
            engine._locate_with_retry(step)


# ---------------------------------------------------------------------------
# State machine — on_failure and retry transitions
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_abort_stops_on_failure(self) -> None:
        """When a step with on_failure=abort fails, subsequent steps are skipped."""
        fail_step = Step(
            id="fail",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="Never"),
            timeout=0.1,
            retries=1,
            on_failure=OnFailure.ABORT,
        )
        next_step = Step(id="after", action=StepAction.WAIT, wait=0.0)
        ocr = MagicMock()
        ocr.find_text.return_value = None
        engine = _make_engine([fail_step, next_step], ocr_engine=ocr)

        with pytest.raises(ScenarioError):
            engine.run()

        step_ids = [r.step_id for r in engine.context.history]
        assert "fail" in step_ids
        assert "after" not in step_ids

    def test_skip_continues_after_failure(self) -> None:
        """When a step with on_failure=skip fails, the next step still runs."""
        fail_step = Step(
            id="fail",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="Never"),
            timeout=0.1,
            retries=1,
            on_failure=OnFailure.SKIP,
        )
        ok_step = Step(id="ok", action=StepAction.WAIT, wait=0.0)
        ocr = MagicMock()
        ocr.find_text.return_value = None
        engine = _make_engine([fail_step, ok_step], ocr_engine=ocr)

        engine.run()  # should not raise

        step_ids = [r.step_id for r in engine.context.history]
        assert "fail" in step_ids
        assert "ok" in step_ids
        assert engine.context.failed_steps[0].step_id == "fail"

    def test_retry_attempts_step_multiple_times(self) -> None:
        """A step with retries=3 is attempted up to 3 times before failing."""
        step = Step(
            id="retry_me",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="Elusive"),
            timeout=0.05,
            retries=3,
            on_failure=OnFailure.ABORT,
        )
        ocr = MagicMock()
        ocr.find_text.return_value = None  # never found
        engine = _make_engine([step], ocr_engine=ocr)

        with pytest.raises(ScenarioError):
            engine.run()

        # Step recorded once (per _execute_step_once call per retry)
        step_records = [r for r in engine.context.history if r.step_id == "retry_me"]
        assert len(step_records) == 3

    def test_retry_succeeds_on_second_attempt(self) -> None:
        """Step succeeds on the second attempt (first fails, second finds element)."""
        step = Step(
            id="retry_ok",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="OK"),
            timeout=0.5,
            retries=3,
            on_failure=OnFailure.ABORT,
        )
        found = _find_result(element="OK")
        ocr = MagicMock()
        # First call returns None, second returns result
        ocr.find_text.side_effect = [None, found, found]
        engine = _make_engine([step], ocr_engine=ocr)

        engine.run()

        step_records = [r for r in engine.context.history if r.step_id == "retry_ok"]
        # First attempt fails (timeout after one poll), second succeeds
        assert any(r.success for r in step_records)

    def test_teardown_runs_even_after_failure(self) -> None:
        """Teardown steps execute even when the main scenario aborts."""
        fail_step = Step(
            id="abort_me",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="Ghost"),
            timeout=0.05,
            retries=1,
            on_failure=OnFailure.ABORT,
        )
        teardown_step = Step(id="cleanup", action=StepAction.WAIT, wait=0.0)

        ocr = MagicMock()
        ocr.find_text.return_value = None
        engine = _make_engine([fail_step], teardown=[teardown_step], ocr_engine=ocr)

        with pytest.raises(ScenarioError):
            engine.run()

        step_ids = [r.step_id for r in engine.context.history]
        assert "cleanup" in step_ids

    def test_screen_mismatch_raises(self) -> None:
        """ScreenMismatch is raised when expect_screen doesn't match."""
        step = Step(
            id="verify",
            action=StepAction.WAIT,
            wait=0.0,
            expect_screen="expected_screen",
            retries=1,
        )
        # Screen analyzer returns a different screen
        analyzer = MagicMock()
        from screenwalker.vision.screen_state import ScreenIdentification
        analyzer.identify.return_value = ScreenIdentification(
            screen_id="wrong_screen", confidence=0.9
        )
        engine = _make_engine([step])
        engine._screen_analyzer = analyzer
        engine.context.dry_run = False

        with pytest.raises(ScenarioError):
            engine.run()

    def test_all_steps_succeed(self) -> None:
        """Multiple steps all succeed and are recorded as successful."""
        steps = [
            Step(id=f"step_{i}", action=StepAction.WAIT, wait=0.0)
            for i in range(5)
        ]
        engine = _make_engine(steps)
        engine.run()

        assert engine.context.step_count == 5
        assert engine.context.failed_steps == []


# ---------------------------------------------------------------------------
# Variable interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_text_interpolated_before_type(self) -> None:
        step = Step(
            id="ty",
            action=StepAction.TYPE,
            text="{{ greeting }}",
        )
        kb = MagicMock()
        engine = _make_engine([step], keyboard=kb)
        engine.context.set_variable("greeting", "Hello World")

        engine.run()

        kb.type_text.assert_called_once_with("Hello World")

    def test_find_query_interpolated(self) -> None:
        step = Step(
            id="find",
            action=StepAction.ASSERT_VISIBLE,
            find=FindSpec(method=FindMethod.OCR, query="{{ label }}"),
            timeout=0.5,
            retries=1,
        )
        ocr = MagicMock()
        ocr.find_text.return_value = _find_result(element="Save")
        engine = _make_engine([step], ocr_engine=ocr)
        engine.context.set_variable("label", "Save")

        engine.run()

        # find_text called with interpolated query
        call_query = ocr.find_text.call_args[0][1]
        assert call_query == "Save"


# ---------------------------------------------------------------------------
# ActionCache
# ---------------------------------------------------------------------------


class TestActionCache:
    def test_lookup_returns_none_when_disabled(self, tmp_path: Path) -> None:
        cache = ActionCache(path=tmp_path / "c.json", enabled=False)
        assert cache.lookup("screen", "element") is None

    def test_store_and_lookup_roundtrip(self, tmp_path: Path) -> None:
        cache = ActionCache(path=tmp_path / "cache.json", ttl=3600)
        result = _find_result(element="OK", x=10, y=20, w=50, h=25, method="ocr")
        cache.store("main_screen", result)
        hit = cache.lookup("main_screen", "OK")
        assert hit is not None
        assert hit.element == "OK"
        assert hit.bbox == BBox(10, 20, 50, 25)

    def test_stale_entry_returns_none(self, tmp_path: Path) -> None:
        import time as _time
        cache = ActionCache(path=tmp_path / "cache.json", ttl=0.001)
        result = _find_result(element="Old")
        cache.store("s", result)
        _time.sleep(0.05)
        assert cache.lookup("s", "Old") is None

    def test_invalidate_removes_entry(self, tmp_path: Path) -> None:
        cache = ActionCache(path=tmp_path / "cache.json", ttl=3600)
        result = _find_result(element="Btn")
        cache.store("screen1", result)
        cache.invalidate("screen1", "Btn")
        assert cache.lookup("screen1", "Btn") is None

    def test_invalidate_all_screen_entries(self, tmp_path: Path) -> None:
        cache = ActionCache(path=tmp_path / "cache.json", ttl=3600)
        cache.store("screen1", _find_result(element="A"))
        cache.store("screen1", _find_result(element="B"))
        cache.store("screen2", _find_result(element="C"))
        cache.invalidate("screen1")
        assert cache.lookup("screen1", "A") is None
        assert cache.lookup("screen1", "B") is None
        assert cache.lookup("screen2", "C") is not None

    def test_evict_stale_removes_expired(self, tmp_path: Path) -> None:
        import time as _time
        cache = ActionCache(path=tmp_path / "cache.json", ttl=0.001)
        cache.store("s", _find_result(element="X"))
        _time.sleep(0.05)
        evicted = cache.evict_stale()
        assert evicted == 1
        assert cache._store == {}

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        c1 = ActionCache(path=path, ttl=3600)
        c1.store("s", _find_result(element="Btn"))
        # New instance loads from disk
        c2 = ActionCache(path=path, ttl=3600)
        assert c2.lookup("s", "Btn") is not None
