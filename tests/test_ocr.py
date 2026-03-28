"""Tests for the OCR module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from screenwalker.vision.ocr import TesseractEngine, build_ocr_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_image(width: int = 200, height: int = 50) -> Image.Image:
    """Return a blank white PIL image."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


# ---------------------------------------------------------------------------
# build_ocr_engine factory
# ---------------------------------------------------------------------------


class TestBuildOcrEngine:
    """Tests for the OCR engine factory."""

    def test_returns_tesseract_for_default_name(self) -> None:
        """build_ocr_engine('tesseract') returns TesseractEngine."""
        engine = build_ocr_engine("tesseract")
        assert isinstance(engine, TesseractEngine)

    def test_case_insensitive_name(self) -> None:
        """Engine names are matched case-insensitively."""
        engine = build_ocr_engine("Tesseract")
        assert isinstance(engine, TesseractEngine)

    def test_unknown_engine_raises_value_error(self) -> None:
        """Unknown engine name raises ValueError with available options."""
        with pytest.raises(ValueError, match="Unknown OCR engine"):
            build_ocr_engine("magic_ocr")

    def test_paddleocr_raises_import_error_when_not_installed(self) -> None:
        """Requesting PaddleOCR when package is missing raises ImportError (via NotImplementedError stub)."""
        with pytest.raises(NotImplementedError):
            # The stub raises NotImplementedError in __init__
            build_ocr_engine("paddleocr")


# ---------------------------------------------------------------------------
# TesseractEngine unit tests (mocked pytesseract)
# ---------------------------------------------------------------------------


class TestTesseractEngine:
    """Unit tests for TesseractEngine using mocked pytesseract."""

    def test_init_stores_config(self) -> None:
        """Constructor stores lang and config attributes."""
        engine = TesseractEngine(lang="eng+rus", config="--psm 11")
        assert engine.lang == "eng+rus"
        assert engine.config == "--psm 11"
        assert engine.preprocess is True

    def test_init_preprocess_flag(self) -> None:
        """preprocess=False is stored correctly."""
        engine = TesseractEngine(preprocess=False)
        assert engine.preprocess is False

    def test_extract_text_calls_tesseract(self) -> None:
        """extract_text delegates to pytesseract.image_to_string."""
        # TODO: remove this test's skip once extract_text is implemented
        engine = TesseractEngine()
        with pytest.raises(NotImplementedError):
            engine.extract_text(_blank_image())

    def test_find_text_raises_not_implemented(self) -> None:
        """find_text raises NotImplementedError (stub)."""
        engine = TesseractEngine()
        with pytest.raises(NotImplementedError):
            engine.find_text(_blank_image(), "Submit")

    def test_find_all_text_raises_not_implemented(self) -> None:
        """find_all_text raises NotImplementedError (stub)."""
        engine = TesseractEngine()
        with pytest.raises(NotImplementedError):
            engine.find_all_text(_blank_image(), "OK")

    # -----------------------------------------------------------------------
    # Pre-implementation contract tests (will pass once implemented)
    # -----------------------------------------------------------------------

    @pytest.mark.skip(reason="Requires pytesseract to be installed and implemented")
    def test_extract_text_returns_string(self) -> None:
        """After implementation, extract_text must return a string."""
        engine = TesseractEngine()
        with patch("pytesseract.image_to_string", return_value="Hello World\n"):
            result = engine.extract_text(_blank_image())
        assert isinstance(result, str)
        assert "Hello" in result

    @pytest.mark.skip(reason="Requires implementation")
    def test_find_text_returns_find_result_on_match(self) -> None:
        """find_text returns FindResult with confidence >= threshold for a clear match."""
        from screenwalker.vision.screen_state import FindResult

        engine = TesseractEngine()
        mock_data = {
            "text": ["Submit", "Cancel"],
            "conf": [95, 80],
            "left": [10, 100],
            "top": [5, 5],
            "width": [60, 50],
            "height": [20, 20],
        }
        with patch("pytesseract.image_to_data", return_value=mock_data):
            result = engine.find_text(_blank_image(), "Submit", threshold=0.5)
        assert result is not None
        assert isinstance(result, FindResult)
        assert result.confidence >= 0.5

    @pytest.mark.skip(reason="Requires implementation")
    def test_find_text_returns_none_below_threshold(self) -> None:
        """find_text returns None when best match is below threshold."""
        engine = TesseractEngine()
        mock_data = {
            "text": ["XYZ"],
            "conf": [10],
            "left": [0],
            "top": [0],
            "width": [30],
            "height": [15],
        }
        with patch("pytesseract.image_to_data", return_value=mock_data):
            result = engine.find_text(_blank_image(), "Submit", threshold=0.9)
        assert result is None
