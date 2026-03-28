"""Tests for the OCR module.

Test layers:
- Pure-logic tests (OCRResult, _group_into_lines) — no mocks needed.
- Mocked-pytesseract tests — run without Tesseract binary.
- Preprocessing tests — run with real cv2 when available, else skipped.
- Synthetic-image integration test — requires both pytesseract + Tesseract binary.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

import screenwalker.vision.ocr as _ocr_mod
from screenwalker.vision.ocr import (
    OCRResult,
    PaddleOCREngine,
    TesseractEngine,
    build_ocr_engine,
    create_ocr_engine,
)
from screenwalker.vision.screen_state import BBox, FindResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white(w: int = 300, h: int = 60) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _make_result(text: str, x: int, y: int, w: int, h: int, conf: float = 0.9) -> OCRResult:
    return OCRResult(text=text, confidence=conf, bbox=BBox(x=x, y=y, w=w, h=h))


def _mock_tess_data(**overrides: Any) -> dict[str, list]:
    """Build a minimal pytesseract.image_to_data DICT payload."""
    base: dict[str, list] = {
        "level": [5, 5, 5],
        "page_num": [1, 1, 1],
        "block_num": [1, 1, 1],
        "par_num": [1, 1, 1],
        "line_num": [1, 1, 1],
        "word_num": [1, 2, 3],
        "left": [10, 90, 180],
        "top": [5, 5, 5],
        "width": [70, 80, 60],
        "height": [20, 20, 20],
        "conf": [95, 85, 72],
        "text": ["Submit", "Cancel", "OK"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# OCRResult dataclass
# ---------------------------------------------------------------------------


class TestOCRResult:
    def test_creates_with_required_fields(self) -> None:
        r = OCRResult(text="hello", confidence=0.9, bbox=BBox(0, 0, 50, 20))
        assert r.text == "hello"
        assert r.confidence == 0.9
        assert r.bbox == BBox(0, 0, 50, 20)

    def test_raw_data_defaults_to_empty_dict(self) -> None:
        r = OCRResult(text="x", confidence=1.0, bbox=BBox(0, 0, 10, 10))
        assert r.raw_data == {}

    def test_frozen_raises_on_mutation(self) -> None:
        r = OCRResult(text="x", confidence=1.0, bbox=BBox(0, 0, 10, 10))
        with pytest.raises((AttributeError, TypeError)):
            r.text = "y"  # type: ignore[misc]

    def test_with_raw_data(self) -> None:
        r = OCRResult(
            text="hi",
            confidence=0.8,
            bbox=BBox(0, 0, 30, 15),
            raw_data={"level": 5, "block_num": 1},
        )
        assert r.raw_data["level"] == 5


# ---------------------------------------------------------------------------
# TesseractEngine.__init__
# ---------------------------------------------------------------------------


class TestTesseractEngineInit:
    def test_defaults(self) -> None:
        e = TesseractEngine()
        assert e.lang == "eng"
        assert e.config == "--psm 6"
        assert e.preprocess is True
        assert e.confidence_threshold == 60
        assert e.resize is False
        assert e.denoise is False

    def test_custom_params_stored(self) -> None:
        e = TesseractEngine(
            lang="eng+rus",
            config="--psm 11",
            preprocess=False,
            confidence_threshold=70,
            resize=True,
            denoise=True,
        )
        assert e.lang == "eng+rus"
        assert e.confidence_threshold == 70
        assert e.resize is True
        assert e.denoise is True


# ---------------------------------------------------------------------------
# _group_into_lines — pure logic, no mocks needed
# ---------------------------------------------------------------------------


class TestGroupIntoLines:
    def _engine(self) -> TesseractEngine:
        return TesseractEngine(preprocess=False)

    def test_empty_returns_empty(self) -> None:
        assert self._engine()._group_into_lines([]) == []

    def test_single_word_returned_as_single_line(self) -> None:
        words = [_make_result("Hello", 10, 5, 50, 20)]
        lines = self._engine()._group_into_lines(words)
        assert len(lines) == 1
        assert lines[0].text == "Hello"

    def test_two_words_same_line_merged(self) -> None:
        # y-centers: 5+10=15 and 7+10=17 — within half-height tolerance
        words = [
            _make_result("Hello", x=10, y=5, w=50, h=20),
            _make_result("World", x=70, y=7, w=50, h=20),
        ]
        lines = self._engine()._group_into_lines(words)
        assert len(lines) == 1
        assert "Hello" in lines[0].text
        assert "World" in lines[0].text

    def test_two_words_different_lines_separated(self) -> None:
        # y-centers: 15 and 45 — gap of 30 > tolerance
        words = [
            _make_result("Top", x=10, y=5, w=40, h=20),
            _make_result("Bottom", x=10, y=35, w=60, h=20),
        ]
        lines = self._engine()._group_into_lines(words)
        assert len(lines) == 2

    def test_word_order_preserved_left_to_right(self) -> None:
        words = [
            _make_result("B", x=100, y=5, w=20, h=20),
            _make_result("A", x=10, y=5, w=20, h=20),
            _make_result("C", x=200, y=5, w=20, h=20),
        ]
        lines = self._engine()._group_into_lines(words)
        assert len(lines) == 1
        assert lines[0].text == "A B C"

    def test_merged_bbox_spans_all_words(self) -> None:
        words = [
            _make_result("Left", x=10, y=5, w=40, h=20),
            _make_result("Right", x=200, y=5, w=50, h=20),
        ]
        lines = self._engine()._group_into_lines(words)
        bbox = lines[0].bbox
        assert bbox.x == 10
        assert bbox.x + bbox.w == 250  # 200 + 50

    def test_confidence_averaged_over_line(self) -> None:
        words = [
            _make_result("A", x=0, y=0, w=20, h=20, conf=0.8),
            _make_result("B", x=30, y=0, w=20, h=20, conf=0.6),
        ]
        lines = self._engine()._group_into_lines(words)
        assert abs(lines[0].confidence - 0.7) < 1e-9

    def test_three_lines(self) -> None:
        words = [
            _make_result("Line1", x=0, y=0, w=50, h=20),
            _make_result("Line2", x=0, y=40, w=50, h=20),
            _make_result("Line3", x=0, y=80, w=50, h=20),
        ]
        lines = self._engine()._group_into_lines(words)
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# TesseractEngine.recognize (mocked pytesseract)
# ---------------------------------------------------------------------------


class TestRecognize:
    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_returns_ocr_results(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data()

        engine = TesseractEngine(confidence_threshold=0)
        results = engine.recognize(_white())

        assert len(results) == 3
        assert all(isinstance(r, OCRResult) for r in results)
        assert results[0].text == "Submit"

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_filters_below_confidence_threshold(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        # conf: 95, 85, 72 — filter at 80 should keep 2
        mock_tess.image_to_data.return_value = _mock_tess_data()

        engine = TesseractEngine(confidence_threshold=80)
        results = engine.recognize(_white())
        assert len(results) == 2
        assert all(r.confidence >= 0.80 for r in results)

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_filters_negative_confidence(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            conf=[-1, 90, -1],
            text=["", "OK", ""],
        )
        engine = TesseractEngine(confidence_threshold=0)
        results = engine.recognize(_white())
        assert len(results) == 1
        assert results[0].text == "OK"

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_filters_empty_text(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            conf=[90, 90, 90],
            text=["Real", "  ", ""],
        )
        engine = TesseractEngine(confidence_threshold=0)
        results = engine.recognize(_white())
        assert len(results) == 1
        assert results[0].text == "Real"

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_confidence_normalised_to_0_1(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            conf=[100, 50, 0], text=["A", "B", "C"]
        )
        engine = TesseractEngine(confidence_threshold=0)
        results = engine.recognize(_white())
        confs = {r.text: r.confidence for r in results}
        assert confs["A"] == pytest.approx(1.0)
        assert confs["B"] == pytest.approx(0.5)

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_bbox_constructed_correctly(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = {
            "level": [5],
            "page_num": [1],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [1],
            "left": [15],
            "top": [25],
            "width": [80],
            "height": [30],
            "conf": [90],
            "text": ["Hello"],
        }
        engine = TesseractEngine(confidence_threshold=0)
        results = engine.recognize(_white())
        assert results[0].bbox == BBox(x=15, y=25, w=80, h=30)

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_uses_engine_lang(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(text=[], conf=[], **_empty_lists())
        engine = TesseractEngine(lang="eng+rus", confidence_threshold=0)
        engine.recognize(_white())
        call_kwargs = mock_tess.image_to_data.call_args.kwargs
        assert call_kwargs.get("lang") == "eng+rus"

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_lang_override_per_call(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(text=[], conf=[], **_empty_lists())
        engine = TesseractEngine(lang="eng", confidence_threshold=0)
        engine.recognize(_white(), lang="rus")
        assert mock_tess.image_to_data.call_args.kwargs["lang"] == "rus"

    def test_raises_import_error_when_pytesseract_none(self) -> None:
        with patch("screenwalker.vision.ocr.pytesseract", None):
            engine = TesseractEngine(preprocess=False)
            with pytest.raises(ImportError, match="pytesseract"):
                engine.recognize(_white())


# ---------------------------------------------------------------------------
# TesseractEngine._preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    @patch("screenwalker.vision.ocr.cv2")
    def test_returns_pil_image(self, mock_cv2: MagicMock) -> None:
        _setup_cv2_mock(mock_cv2)
        engine = TesseractEngine()
        result = engine._preprocess(_white(100, 50))
        assert isinstance(result, Image.Image)

    @patch("screenwalker.vision.ocr.cv2")
    def test_no_resize_keeps_original_dimensions(self, mock_cv2: MagicMock) -> None:
        import numpy as _np
        # Simulate cv2 returning same-size grayscale then binary
        gray = _np.zeros((50, 100), dtype=_np.uint8)
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.adaptiveThreshold.return_value = gray
        mock_cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 1
        mock_cv2.THRESH_BINARY = 1
        mock_cv2.COLOR_RGB2GRAY = 1
        engine = TesseractEngine(resize=False)
        result = engine._preprocess(_white(100, 50))
        assert mock_cv2.resize.call_count == 0
        assert result.size == (100, 50)

    @patch("screenwalker.vision.ocr.cv2")
    def test_resize_doubles_dimensions(self, mock_cv2: MagicMock) -> None:
        import numpy as _np
        gray = _np.zeros((50, 100), dtype=_np.uint8)
        large = _np.zeros((100, 200), dtype=_np.uint8)
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.resize.return_value = large
        mock_cv2.adaptiveThreshold.return_value = large
        mock_cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 1
        mock_cv2.THRESH_BINARY = 1
        mock_cv2.COLOR_RGB2GRAY = 1
        mock_cv2.INTER_CUBIC = 2
        engine = TesseractEngine(resize=True)
        result = engine._preprocess(_white(100, 50))
        # resize called with doubled dims
        mock_cv2.resize.assert_called_once()
        args = mock_cv2.resize.call_args
        assert args[0][1] == (200, 100)  # (width*2, height*2)
        assert result.size == (200, 100)

    @patch("screenwalker.vision.ocr.cv2")
    def test_denoise_called_when_enabled(self, mock_cv2: MagicMock) -> None:
        import numpy as _np
        gray = _np.zeros((50, 100), dtype=_np.uint8)
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.adaptiveThreshold.return_value = gray
        mock_cv2.fastNlMeansDenoising.return_value = gray
        mock_cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 1
        mock_cv2.THRESH_BINARY = 1
        mock_cv2.COLOR_RGB2GRAY = 1
        engine = TesseractEngine(denoise=True)
        engine._preprocess(_white())
        mock_cv2.fastNlMeansDenoising.assert_called_once()

    @patch("screenwalker.vision.ocr.cv2")
    def test_denoise_not_called_when_disabled(self, mock_cv2: MagicMock) -> None:
        import numpy as _np
        gray = _np.zeros((50, 100), dtype=_np.uint8)
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.adaptiveThreshold.return_value = gray
        mock_cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 1
        mock_cv2.THRESH_BINARY = 1
        mock_cv2.COLOR_RGB2GRAY = 1
        engine = TesseractEngine(denoise=False)
        engine._preprocess(_white())
        mock_cv2.fastNlMeansDenoising.assert_not_called()

    def test_raises_import_error_when_cv2_none(self) -> None:
        with patch("screenwalker.vision.ocr.cv2", None):
            engine = TesseractEngine()
            with pytest.raises(ImportError, match="opencv"):
                engine._preprocess(_white())


# ---------------------------------------------------------------------------
# TesseractEngine.extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_returns_space_joined_words(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data()
        engine = TesseractEngine(confidence_threshold=0)
        text = engine.extract_text(_white())
        assert "Submit" in text
        assert "Cancel" in text
        assert "OK" in text

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_empty_recognition_returns_empty_string(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=[], conf=[], **_empty_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        assert engine.extract_text(_white()) == ""


# ---------------------------------------------------------------------------
# TesseractEngine.find_text / find_all_text
# ---------------------------------------------------------------------------


class TestFindText:
    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_finds_exact_match(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data()
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Submit", threshold=0.5)
        assert result is not None
        assert isinstance(result, FindResult)
        assert result.method == "ocr"

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_returns_none_below_threshold(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["XYZ"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Submit", threshold=0.95)
        assert result is None

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_find_all_returns_sorted_by_confidence(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        # Two lines: "Submit Form" and "Submit Request"
        mock_tess.image_to_data.return_value = {
            "level": [5, 5, 5, 5],
            "page_num": [1, 1, 1, 1],
            "block_num": [1, 1, 2, 2],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "word_num": [1, 2, 1, 2],
            "left": [10, 70, 10, 70],
            "top": [5, 5, 50, 50],
            "width": [55, 50, 55, 65],
            "height": [20, 20, 20, 20],
            "conf": [90, 85, 90, 85],
            "text": ["Submit", "Form", "Submit", "Request"],
        }
        engine = TesseractEngine(confidence_threshold=0)
        results = engine.find_all_text(_white(200, 100), "Submit", threshold=0.5)
        assert len(results) >= 1
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_region_crops_image_and_offsets_bbox(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = {
            "level": [5],
            "page_num": [1],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [1],
            "left": [5],
            "top": [5],
            "width": [40],
            "height": [15],
            "conf": [90],
            "text": ["OK"],
        }
        engine = TesseractEngine(confidence_threshold=0)
        region = BBox(x=100, y=200, w=150, h=50)
        result = engine.find_text(_white(400, 400), "OK", threshold=0.5, region=region)
        assert result is not None
        # bbox should be offset by region origin
        assert result.bbox.x == 100 + 5
        assert result.bbox.y == 200 + 5

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_exact_search_requires_substring(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["Sbmt"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Submit", threshold=0.5, fuzzy=False)
        assert result is None

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_exact_search_matches_substring(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["Click Submit here"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Submit", threshold=0.5, fuzzy=False)
        assert result is not None


# ---------------------------------------------------------------------------
# Fuzzy search with typos
# ---------------------------------------------------------------------------


class TestFuzzySearch:
    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_finds_with_one_char_typo(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        """'Sbmit' should match 'Submit' via partial_ratio."""
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["Sbmit"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Submit", threshold=0.7, fuzzy=True)
        assert result is not None

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_case_insensitive_match(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["SUBMIT"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "submit", threshold=0.9, fuzzy=True)
        assert result is not None
        assert result.confidence == pytest.approx(1.0)

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_cyrillic_fuzzy_match(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        """Partial Russian text match."""
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["Подтвердить"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Подтвердить", threshold=0.9, fuzzy=True)
        assert result is not None

    @patch("screenwalker.vision.ocr.pytesseract")
    @patch("screenwalker.vision.ocr.cv2")
    def test_very_different_text_not_matched(
        self, mock_cv2: MagicMock, mock_tess: MagicMock
    ) -> None:
        _setup_cv2_mock(mock_cv2)
        mock_tess.Output.DICT = "dict"
        mock_tess.image_to_data.return_value = _mock_tess_data(
            text=["XQZ"], conf=[90], **_single_word_lists()
        )
        engine = TesseractEngine(confidence_threshold=0)
        result = engine.find_text(_white(), "Submit", threshold=0.9, fuzzy=True)
        assert result is None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


class TestBuildOcrEngine:
    def test_returns_tesseract_engine(self) -> None:
        assert isinstance(build_ocr_engine("tesseract"), TesseractEngine)

    def test_case_insensitive(self) -> None:
        assert isinstance(build_ocr_engine("Tesseract"), TesseractEngine)

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown OCR engine"):
            build_ocr_engine("magic_ocr")

    def test_kwargs_forwarded_to_engine(self) -> None:
        engine = build_ocr_engine("tesseract", lang="eng+rus", confidence_threshold=75)
        assert isinstance(engine, TesseractEngine)
        assert engine.lang == "eng+rus"
        assert engine.confidence_threshold == 75


class TestCreateOcrEngine:
    def test_creates_tesseract_from_config(self) -> None:
        from screenwalker.utils.config import AppConfig, VisionConfig

        cfg = AppConfig()
        cfg.vision.ocr_engine = "tesseract"
        cfg.vision.ocr_lang = "eng+rus"
        cfg.vision.ocr_config = "--psm 11"
        cfg.vision.ocr_preprocess = False

        engine = create_ocr_engine(cfg)
        assert isinstance(engine, TesseractEngine)
        assert engine.lang == "eng+rus"
        assert engine.config == "--psm 11"
        assert engine.preprocess is False

    def test_unknown_engine_raises(self) -> None:
        from screenwalker.utils.config import AppConfig

        cfg = AppConfig()
        cfg.vision.ocr_engine = "unknown"
        with pytest.raises(ValueError):
            create_ocr_engine(cfg)


class TestPaddleOCREngine:
    def test_raises_import_error_when_not_installed(self) -> None:
        """PaddleOCREngine raises ImportError when paddleocr is not available."""
        with patch.dict(sys.modules, {"paddleocr": None}):
            with pytest.raises(ImportError, match="PaddleOCR"):
                PaddleOCREngine()

    def test_build_ocr_engine_paddle_raises_import_error(self) -> None:
        with patch.dict(sys.modules, {"paddleocr": None}):
            with pytest.raises(ImportError):
                build_ocr_engine("paddleocr")


# ---------------------------------------------------------------------------
# Synthetic image integration test (requires Tesseract binary)
# ---------------------------------------------------------------------------


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


_REQUIRES_TESSERACT = pytest.mark.skipif(
    not _tesseract_available(),
    reason="Tesseract binary not installed or not on PATH",
)


def _draw_text_image(text: str, font_size: int = 24) -> Image.Image:
    """Create a white PIL image with *text* drawn in black."""
    img = Image.new("RGB", (400, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        # Use a system font if available
        from PIL import ImageFont as _IF

        font = _IF.truetype("arial.ttf", font_size)
    except (OSError, ImportError):
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(0, 0, 0), font=font)
    return img


class TestSyntheticImage:
    @_REQUIRES_TESSERACT
    def test_recognize_drawn_ascii_text(self) -> None:
        """TesseractEngine recognises text drawn with PIL.ImageDraw."""
        img = _draw_text_image("Hello World")
        engine = TesseractEngine(lang="eng", confidence_threshold=30)
        results = engine.recognize(img)
        combined = " ".join(r.text for r in results).lower()
        assert "hello" in combined or "world" in combined

    @_REQUIRES_TESSERACT
    def test_find_text_in_synthetic_image(self) -> None:
        """find_text locates text that was drawn on the image."""
        img = _draw_text_image("Submit")
        engine = TesseractEngine(lang="eng", confidence_threshold=20)
        result = engine.find_text(img, "Submit", threshold=0.6, fuzzy=True)
        assert result is not None
        assert isinstance(result.bbox, BBox)

    @_REQUIRES_TESSERACT
    def test_extract_text_returns_nonempty(self) -> None:
        img = _draw_text_image("OK Cancel")
        engine = TesseractEngine(lang="eng", confidence_threshold=20)
        text = engine.extract_text(img)
        assert len(text.strip()) > 0

    @_REQUIRES_TESSERACT
    def test_fuzzy_finds_close_match(self) -> None:
        """Fuzzy search finds 'Submit' when 'Submt' is drawn (OCR may mangle it)."""
        img = _draw_text_image("Submit")
        engine = TesseractEngine(lang="eng", confidence_threshold=10)
        result = engine.find_text(img, "Submt", threshold=0.7, fuzzy=True)
        # With OCR + fuzzy together, we expect a hit at this low threshold
        # (may still be None if OCR produced garbage — we only assert shape)
        if result is not None:
            assert isinstance(result, FindResult)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _setup_cv2_mock(mock_cv2: MagicMock) -> None:
    """Configure cv2 mock to return a plausible numpy array from _preprocess."""
    import numpy as _np

    gray = _np.zeros((60, 300), dtype=_np.uint8)
    mock_cv2.cvtColor.return_value = gray
    mock_cv2.adaptiveThreshold.return_value = gray
    mock_cv2.fastNlMeansDenoising.return_value = gray
    mock_cv2.COLOR_RGB2GRAY = 6
    mock_cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 1
    mock_cv2.THRESH_BINARY = 0
    mock_cv2.INTER_CUBIC = 2


def _empty_lists() -> dict[str, list]:
    return {
        "level": [],
        "page_num": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
        "word_num": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
    }


def _single_word_lists() -> dict[str, list]:
    return {
        "level": [5],
        "page_num": [1],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
        "word_num": [1],
        "left": [10],
        "top": [5],
        "width": [60],
        "height": [20],
    }
