"""Tests for recovery actions and wait strategies."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from screenwalker.core.context import RunContext
from screenwalker.core.engine import ScenarioEngine, _RecoveryGoto
from screenwalker.core.errors import ElementNotFound, ScenarioError, ScreenMismatch, StepTimeout
from screenwalker.core.step import (
    FindMethod,
    FindSpec,
    OnFailure,
    RecoveryAction,
    RecoveryTrigger,
    Step,
    StepAction,
)
from screenwalker.utils.config import AppConfig
from screenwalker.vision.screen_state import BBox, FindResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (800, 600), color=color)


def _find_result(element: str = "OK") -> FindResult:
    return FindResult(element=element, confidence=0.9, bbox=BBox(100, 200, 80, 30), method="ocr")


def _make_engine(
    steps: list[Step],
    ocr_engine=None,
    capture=None,
    teardown: list[Step] | None = None,
) -> ScenarioEngine:
    cfg = AppConfig()
    cfg.timeouts.step_default = 0.2
    cfg.timeouts.poll_interval = 0.05
    cfg.timeouts.screenshot_delay = 0.0
    cfg.retry.backoff_base = 0.0
    cfg.retry.backoff_max = 0.0

    ctx = RunContext("test")

    if capture is None:
        capture = MagicMock()
        capture.capture_full.return_value = _make_image()

    return ScenarioEngine(
        scenario_name="test",
        steps=steps,
        teardown_steps=teardown or [],
        context=ctx,
        config=cfg,
        capture=capture,
        ocr_engine=ocr_engine,
        mouse=MagicMock(),
        keyboard=MagicMock(),
        clipboard=MagicMock(),
        cache=MagicMock(lookup=MagicMock(return_value=None)),
        step_logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# _RecoveryGoto
# ---------------------------------------------------------------------------


class TestRecoveryGoto:
    def test_str_contains_target(self) -> None:
        exc = _RecoveryGoto("main_menu")
        assert "main_menu" in str(exc)

    def test_step_id_attribute(self) -> None:
        exc = _RecoveryGoto("step_42")
        assert exc.step_id == "step_42"


# ---------------------------------------------------------------------------
# _apply_recovery
# ---------------------------------------------------------------------------


class TestApplyRecovery:
    def test_no_matching_trigger_returns_true(self) -> None:
        step = Step(
            id="s",
            action=StepAction.WAIT,
            recovery=[
                RecoveryAction(trigger=RecoveryTrigger.TIMEOUT, action="scroll_down")
            ],
        )
        engine = _make_engine([step])
        result = engine._apply_recovery(step, RecoveryTrigger.ELEMENT_NOT_FOUND)
        assert result is True

    def test_screenshot_and_abort_returns_false(self, tmp_path: Path) -> None:
        step = Step(
            id="s",
            action=StepAction.WAIT,
            recovery=[
                RecoveryAction(
                    trigger=RecoveryTrigger.TIMEOUT,
                    action="screenshot_and_abort",
                )
            ],
        )
        cfg = AppConfig()
        ctx = RunContext("test", output_dir=tmp_path)
        capture = MagicMock()
        capture.capture_full.return_value = _make_image()
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
        engine.context.update_screenshot(_make_image())
        result = engine._apply_recovery(step, RecoveryTrigger.TIMEOUT)
        assert result is False

    def test_scroll_down_calls_pyautogui(self) -> None:
        step = Step(
            id="s",
            action=StepAction.WAIT,
            recovery=[
                RecoveryAction(trigger=RecoveryTrigger.ELEMENT_NOT_FOUND, action="scroll_down")
            ],
        )
        engine = _make_engine([step])
        with patch("pyautogui.scroll") as mock_scroll, patch("time.sleep"):
            result = engine._apply_recovery(step, RecoveryTrigger.ELEMENT_NOT_FOUND)
        assert result is True
        mock_scroll.assert_called_once_with(-3)

    def test_press_escape_calls_pyautogui(self) -> None:
        step = Step(
            id="s",
            action=StepAction.WAIT,
            recovery=[
                RecoveryAction(trigger=RecoveryTrigger.SCREEN_MISMATCH, action="press_escape")
            ],
        )
        engine = _make_engine([step])
        with patch("pyautogui.press") as mock_press, patch("time.sleep"):
            result = engine._apply_recovery(step, RecoveryTrigger.SCREEN_MISMATCH)
        assert result is True
        mock_press.assert_called_once_with("escape")

    def test_press_key_calls_pyautogui_hotkey(self) -> None:
        step = Step(
            id="s",
            action=StepAction.WAIT,
            recovery=[
                RecoveryAction(
                    trigger=RecoveryTrigger.ELEMENT_NOT_FOUND,
                    action="press_key",
                    keys=["ctrl", "Home"],
                )
            ],
        )
        engine = _make_engine([step])
        with patch("pyautogui.hotkey") as mock_hotkey, patch("time.sleep"):
            result = engine._apply_recovery(step, RecoveryTrigger.ELEMENT_NOT_FOUND)
        assert result is True
        mock_hotkey.assert_called_once_with("ctrl", "Home")

    def test_goto_raises_recovery_goto(self) -> None:
        step = Step(
            id="s",
            action=StepAction.WAIT,
            recovery=[
                RecoveryAction(
                    trigger=RecoveryTrigger.SCREEN_MISMATCH,
                    action="press_escape",
                    then="goto:main_menu",
                )
            ],
        )
        engine = _make_engine([step])
        with patch("pyautogui.press"), patch("time.sleep"):
            with pytest.raises(_RecoveryGoto) as exc_info:
                engine._apply_recovery(step, RecoveryTrigger.SCREEN_MISMATCH)
        assert exc_info.value.step_id == "main_menu"


# ---------------------------------------------------------------------------
# Recovery actions integrated with engine step execution
# ---------------------------------------------------------------------------


class TestRecoveryInEngine:
    def test_scroll_down_recovery_on_element_not_found(self) -> None:
        """On element_not_found, scroll_down is called, then step retries."""
        ocr = MagicMock()
        found = _find_result("Submit")
        # First two calls fail, third succeeds (after scroll recovery)
        ocr.find_text.side_effect = [None, None, found, found]

        step = Step(
            id="click_submit",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="Submit"),
            timeout=0.1,
            retries=3,
            recovery=[
                RecoveryAction(
                    trigger=RecoveryTrigger.ELEMENT_NOT_FOUND,
                    action="scroll_down",
                )
            ],
        )
        engine = _make_engine([step], ocr_engine=ocr)
        with patch("pyautogui.scroll"), patch("time.sleep"):
            engine.run()

        assert engine.context.failed_steps == []

    def test_screenshot_and_abort_stops_retries(self) -> None:
        """screenshot_and_abort causes immediate abort without exhausting retries."""
        ocr = MagicMock()
        ocr.find_text.return_value = None

        step = Step(
            id="find_x",
            action=StepAction.CLICK,
            find=FindSpec(method=FindMethod.OCR, query="X"),
            timeout=0.1,
            retries=5,
            on_failure=OnFailure.ABORT,
            recovery=[
                RecoveryAction(
                    trigger=RecoveryTrigger.TIMEOUT,
                    action="screenshot_and_abort",
                )
            ],
        )
        engine = _make_engine([step], ocr_engine=ocr)
        with pytest.raises(ScenarioError):
            engine.run()

        # Only one attempt recorded (aborted immediately)
        records = [r for r in engine.context.history if r.step_id == "find_x"]
        assert len(records) == 1

    def test_goto_recovery_jumps_to_target_step(self) -> None:
        """After screen mismatch recovery with goto, execution jumps to target step."""
        analyzer = MagicMock()
        from screenwalker.vision.screen_state import ScreenIdentification

        # First call returns wrong screen, then right screen for step_b
        analyzer.identify.side_effect = [
            ScreenIdentification(screen_id="wrong", confidence=0.9),
            ScreenIdentification(screen_id="main", confidence=0.9),
        ]

        step_a = Step(
            id="step_a",
            action=StepAction.WAIT,
            wait=0.0,
            expect_screen="main",
            retries=2,
            recovery=[
                RecoveryAction(
                    trigger=RecoveryTrigger.SCREEN_MISMATCH,
                    action="press_escape",
                    then="goto:step_b",
                )
            ],
        )
        step_b = Step(id="step_b", action=StepAction.WAIT, wait=0.0)

        engine = _make_engine([step_a, step_b])
        engine._screen_analyzer = analyzer

        with patch("pyautogui.press"), patch("time.sleep"):
            engine.run()

        step_ids = [r.step_id for r in engine.context.history]
        assert "step_b" in step_ids


# ---------------------------------------------------------------------------
# Wait strategies
# ---------------------------------------------------------------------------


class TestWaitForScreenChange:
    def test_returns_true_when_screen_changes(self) -> None:
        capture = MagicMock()
        capture.capture_full.side_effect = [
            _make_image((100, 100, 100)),  # initial
            _make_image((200, 200, 200)),  # after change
        ]
        engine = _make_engine([], capture=capture)
        result = engine.wait_for_screen_change(timeout=1.0, poll_interval=0.01, mse_threshold=10.0)
        assert result is True

    def test_returns_false_on_timeout_with_unchanged_screen(self) -> None:
        capture = MagicMock()
        # Always returns same image
        capture.capture_full.return_value = _make_image((128, 128, 128))
        engine = _make_engine([], capture=capture)
        result = engine.wait_for_screen_change(timeout=0.1, poll_interval=0.05, mse_threshold=50.0)
        assert result is False

    def test_uses_config_threshold_by_default(self) -> None:
        capture = MagicMock()
        capture.capture_full.side_effect = [
            _make_image((0, 0, 0)),
            _make_image((255, 255, 255)),  # high MSE
        ]
        engine = _make_engine([], capture=capture)
        engine.config.vision.screen_change_mse_threshold = 1000.0
        result = engine.wait_for_screen_change(timeout=1.0, poll_interval=0.01)
        assert result is True

    def test_returns_false_when_mse_below_threshold(self) -> None:
        # Always returns the same image → MSE is 0, well below any threshold
        capture = MagicMock()
        capture.capture_full.return_value = _make_image((128, 128, 128))
        engine = _make_engine([], capture=capture)
        result = engine.wait_for_screen_change(timeout=0.05, poll_interval=0.02, mse_threshold=1e9)
        assert result is False


class TestWaitForElement:
    def test_returns_result_when_element_found(self) -> None:
        ocr = MagicMock()
        found = _find_result("OK")
        ocr.find_text.return_value = found

        spec = FindSpec(method=FindMethod.OCR, query="OK")
        engine = _make_engine([], ocr_engine=ocr)
        result = engine.wait_for_element(spec, timeout=1.0, poll_interval=0.01)
        assert result is not None
        assert result.element == "OK"

    def test_returns_none_on_timeout(self) -> None:
        ocr = MagicMock()
        ocr.find_text.return_value = None

        spec = FindSpec(method=FindMethod.OCR, query="Missing")
        engine = _make_engine([], ocr_engine=ocr)
        result = engine.wait_for_element(spec, timeout=0.1, poll_interval=0.05)
        assert result is None

    def test_returns_result_after_initial_misses(self) -> None:
        ocr = MagicMock()
        found = _find_result("Button")
        ocr.find_text.side_effect = [None, None, found]

        spec = FindSpec(method=FindMethod.OCR, query="Button")
        engine = _make_engine([], ocr_engine=ocr)
        result = engine.wait_for_element(spec, timeout=1.0, poll_interval=0.01)
        assert result is not None


class TestWaitForElementGone:
    def test_returns_true_when_element_disappears(self) -> None:
        ocr = MagicMock()
        found = _find_result("Loading")
        ocr.find_text.side_effect = [found, found, None]  # disappears on third check

        spec = FindSpec(method=FindMethod.OCR, query="Loading")
        engine = _make_engine([], ocr_engine=ocr)
        result = engine.wait_for_element_gone(spec, timeout=1.0, poll_interval=0.01)
        assert result is True

    def test_returns_false_on_timeout(self) -> None:
        ocr = MagicMock()
        ocr.find_text.return_value = _find_result("Loading")

        spec = FindSpec(method=FindMethod.OCR, query="Loading")
        engine = _make_engine([], ocr_engine=ocr)
        result = engine.wait_for_element_gone(spec, timeout=0.1, poll_interval=0.05)
        assert result is False

    def test_returns_true_immediately_if_already_gone(self) -> None:
        ocr = MagicMock()
        ocr.find_text.return_value = None

        spec = FindSpec(method=FindMethod.OCR, query="Gone")
        engine = _make_engine([], ocr_engine=ocr)
        result = engine.wait_for_element_gone(spec, timeout=1.0, poll_interval=0.01)
        assert result is True
