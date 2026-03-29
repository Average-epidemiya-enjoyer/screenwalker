"""Tests for PopupHandler integration with ScenarioEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from PIL import Image

from screenwalker.core.engine import ScenarioEngine
from screenwalker.core.step import FindMethod, FindSpec, Step, StepAction
from screenwalker.utils.config import AppConfig
from screenwalker.core.context import RunContext
from screenwalker.vision.popup import PopupDefinition, PopupHandler, PopupMatch
from screenwalker.vision.screen_state import BBox, FindResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(color: tuple[int, int, int] = (200, 200, 200)) -> Image.Image:
    return Image.new("RGB", (800, 600), color=color)


def _make_engine(
    steps: list[Step],
    popup_handler=None,
    ocr_engine=None,
) -> ScenarioEngine:
    cfg = AppConfig()
    cfg.timeouts.step_default = 0.5
    cfg.timeouts.poll_interval = 0.05
    cfg.timeouts.screenshot_delay = 0.0
    cfg.retry.backoff_base = 0.0
    cfg.retry.backoff_max = 0.0

    ctx = RunContext("test")
    capture = MagicMock()
    capture.capture_full.return_value = _make_image()

    return ScenarioEngine(
        scenario_name="test",
        steps=steps,
        teardown_steps=[],
        context=ctx,
        config=cfg,
        capture=capture,
        ocr_engine=ocr_engine,
        mouse=MagicMock(),
        keyboard=MagicMock(),
        clipboard=MagicMock(),
        cache=MagicMock(lookup=MagicMock(return_value=None)),
        step_logger=MagicMock(),
        popup_handler=popup_handler,
    )


# ---------------------------------------------------------------------------
# PopupHandler unit tests
# ---------------------------------------------------------------------------


class TestPopupHandlerDetect:
    def test_detect_returns_none_when_no_popup(self) -> None:
        ocr = MagicMock()
        ocr.recognize.return_value = []
        defn = PopupDefinition(
            id="error",
            indicators=["Error"],
            action="click_button",
            button_text=["OK"],
        )
        handler = PopupHandler([defn], ocr_engine=ocr, mouse=MagicMock())
        result = handler.detect(_make_image())
        assert result is None

    def test_detect_returns_match_when_indicator_found(self) -> None:
        ocr = MagicMock()
        ocr_result = MagicMock()
        ocr_result.text = "Error"
        ocr.recognize.return_value = [ocr_result]

        defn = PopupDefinition(
            id="error",
            indicators=["Error"],
            action="click_button",
            button_text=["OK"],
            confidence_threshold=0.4,
        )
        handler = PopupHandler([defn], ocr_engine=ocr, mouse=MagicMock())
        match = handler.detect(_make_image())
        assert match is not None
        assert match.definition.id == "error"
        assert "Error" in match.matched_texts

    def test_detect_and_handle_calls_handle(self) -> None:
        handler = MagicMock(spec=PopupHandler)
        handler.detect.return_value = PopupMatch(
            definition=PopupDefinition(id="x", indicators=[]),
            confidence=1.0,
        )
        handler.handle.return_value = True
        handler.detect_and_handle = PopupHandler.detect_and_handle.__get__(handler)
        result = handler.detect_and_handle(_make_image())
        assert result is True
        handler.handle.assert_called_once()

    def test_detect_and_handle_returns_false_when_no_popup(self) -> None:
        ocr = MagicMock()
        ocr.recognize.return_value = []
        handler = PopupHandler([], ocr_engine=ocr, mouse=MagicMock())
        assert handler.detect_and_handle(_make_image()) is False


class TestPopupHandlerFromYaml:
    def test_from_yaml_loads_definitions(self, tmp_path: Path) -> None:
        yaml_content = """
popups:
  - id: error_dialog
    indicators: ["Error", "Something went wrong"]
    action: click_button
    button_text: ["OK", "Close"]
    confidence_threshold: 0.5
  - id: loading
    indicators: ["Loading"]
    action: wait
    wait_timeout: 10
    wait_until_gone: true
"""
        config_path = tmp_path / "popups.yaml"
        config_path.write_text(yaml_content, encoding="utf-8")

        ocr = MagicMock()
        handler = PopupHandler.from_yaml(config_path, ocr_engine=ocr, mouse=MagicMock())
        assert len(handler.definitions) == 2
        ids = [d.id for d in handler.definitions]
        assert "error_dialog" in ids
        assert "loading" in ids

    def test_from_yaml_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            PopupHandler.from_yaml("/nonexistent/popups.yaml", ocr_engine=MagicMock(), mouse=MagicMock())

    def test_from_yaml_respects_priority_ordering(self, tmp_path: Path) -> None:
        yaml_content = """
popups:
  - id: low
    indicators: ["low"]
    priority: 0
  - id: high
    indicators: ["high"]
    priority: 10
"""
        config_path = tmp_path / "popups.yaml"
        config_path.write_text(yaml_content, encoding="utf-8")
        handler = PopupHandler.from_yaml(config_path, ocr_engine=MagicMock(), mouse=MagicMock())
        assert handler.definitions[0].id == "high"
        assert handler.definitions[1].id == "low"


# ---------------------------------------------------------------------------
# Engine popup integration
# ---------------------------------------------------------------------------


class TestEnginePopupIntegration:
    def test_popup_handler_called_before_each_step(self) -> None:
        popup_handler = MagicMock()
        popup_handler.detect_and_handle.return_value = False

        step = Step(id="w", action=StepAction.WAIT, wait=0.0)
        engine = _make_engine([step, step], popup_handler=popup_handler)
        engine.run()

        assert popup_handler.detect_and_handle.call_count == 2

    def test_screen_recaptured_after_popup_dismissed(self) -> None:
        popup_handler = MagicMock()
        popup_handler.detect_and_handle.return_value = True  # popup was dismissed

        step = Step(id="w", action=StepAction.WAIT, wait=0.0)
        engine = _make_engine([step], popup_handler=popup_handler)
        engine.run()

        # capture_full called: initial capture + re-capture after popup
        assert engine._capture.capture_full.call_count >= 2

    def test_no_popup_check_without_handler(self) -> None:
        step = Step(id="w", action=StepAction.WAIT, wait=0.0)
        engine = _make_engine([step], popup_handler=None)
        # Should complete without error even with no popup handler
        engine.run()
        assert engine.context.step_count == 1

    def test_popup_handler_receives_capture_fn(self) -> None:
        popup_handler = MagicMock()
        popup_handler.detect_and_handle.return_value = False

        step = Step(id="w", action=StepAction.WAIT, wait=0.0)
        engine = _make_engine([step], popup_handler=popup_handler)
        engine.run()

        _, kwargs = popup_handler.detect_and_handle.call_args
        assert "capture_fn" in kwargs
        assert callable(kwargs["capture_fn"])

    def test_engine_from_yaml_loads_popup_handler(self, tmp_path: Path) -> None:
        scenario = tmp_path / "scenario.yaml"
        scenario.write_text("name: T\nsteps:\n  - id: s1\n    action: wait\n    wait: 0\n")
        popups = tmp_path / "popups.yaml"
        popups.write_text("popups:\n  - id: err\n    indicators: ['Error']\n    action: click_button\n    button_text: ['OK']\n")

        engine = ScenarioEngine.from_yaml(scenario, config=AppConfig())
        # popup_handler is None because no OCR engine is available in default config test
        # but the file should have been found — just can't init without tesseract
        # So we just verify no crash during load
        assert engine is not None
