"""Tests for screenwalker.vision.screen_state.

All tests use mocked OCR engines / template matchers — no real binaries needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from screenwalker.vision.screen_state import (
    BBox,
    FindResult,
    PopupInfo,
    ScreenFingerprint,
    ScreenIdentification,
    ScreenStateAnalyzer,
    ScreenStateDetector,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _blank_image() -> Image.Image:
    return Image.new("RGB", (800, 600), color=(240, 240, 240))


@dataclass
class FakeOCRResult:
    """Minimal stand-in for OCRResult without importing the real module."""

    text: str
    confidence: float
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 10, 10))
    raw_data: dict = field(default_factory=dict)


def _mock_ocr(*words: str, confidence: float = 0.95) -> MagicMock:
    """Return a mock OCR engine that returns the given words."""
    results = [FakeOCRResult(text=w, confidence=confidence) for w in words]
    engine = MagicMock()
    engine.recognize.return_value = results
    return engine


def _mock_matcher(found: dict[str, bool] | None = None) -> MagicMock:
    """Return a mock TemplateMatcher.  found maps template_name → bool."""
    found = found or {}
    matcher = MagicMock()

    def _find_one(screenshot, name, threshold=0.8):
        if found.get(name):
            return MagicMock()  # truthy result
        return None

    matcher.find_one.side_effect = _find_one
    return matcher


# ── ScreenFingerprint ─────────────────────────────────────────────────────────


class TestScreenFingerprint:
    def test_defaults(self):
        fp = ScreenFingerprint(screen_id="test")
        assert fp.required_texts == []
        assert fp.forbidden_texts == []
        assert fp.required_templates == []
        assert fp.optional_texts == []
        assert fp.match_threshold == 0.7

    def test_custom_fields(self):
        fp = ScreenFingerprint(
            screen_id="login",
            required_texts=["Sign In"],
            forbidden_texts=["Logout"],
            optional_texts=["Forgot Password"],
            match_threshold=0.8,
        )
        assert fp.screen_id == "login"
        assert fp.required_texts == ["Sign In"]
        assert fp.match_threshold == 0.8

    def test_threshold_validation(self):
        with pytest.raises(Exception):
            ScreenFingerprint(screen_id="bad", match_threshold=1.5)
        with pytest.raises(Exception):
            ScreenFingerprint(screen_id="bad", match_threshold=-0.1)


# ── ScreenStateAnalyzer — scoring ─────────────────────────────────────────────


class TestScoring:
    """Unit-tests for the internal _score() method."""

    def _analyzer(self, *words: str, **kwargs) -> ScreenStateAnalyzer:
        return ScreenStateAnalyzer(ocr_engine=_mock_ocr(*words), **kwargs)

    def test_all_required_found_gives_max_score(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Notepad", "File"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("Notepad", "File"))
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, matched, templates = az._score(fp, img, texts)
        assert conf == pytest.approx(1.0)
        assert "Notepad" in matched
        assert "File" in matched

    def test_missing_required_text_penalises_score(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Notepad", "Missing"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("Notepad"))
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, matched, _ = az._score(fp, img, texts)
        # raw = 1 - 1 = 0, max = 2 → conf = 0
        assert conf == pytest.approx(0.0)

    def test_forbidden_text_penalises(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Notepad"],
            forbidden_texts=["Error"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("Notepad", "Error"))
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, matched, _ = az._score(fp, img, texts)
        # raw = 1 - 1 = 0, max = 1 → conf = 0
        assert conf == pytest.approx(0.0)

    def test_optional_texts_add_bonus(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Notepad"],
            optional_texts=["File", "Edit"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("Notepad", "File", "Edit"))
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, matched, _ = az._score(fp, img, texts)
        # raw = 1 + 0.3 + 0.3 = 1.6, max = 1 + 0.6 = 1.6 → conf = 1.0
        assert conf == pytest.approx(1.0)
        assert "File" in matched
        assert "Edit" in matched

    def test_template_match_adds_score(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Notepad"],
            required_templates=["toolbar"],
            match_threshold=0.0,
        )
        matcher = _mock_matcher({"toolbar": True})
        az = ScreenStateAnalyzer(
            ocr_engine=_mock_ocr("Notepad"), template_matcher=matcher
        )
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, _, templates = az._score(fp, img, texts)
        # raw = 1 + 0.5 = 1.5, max = 1 + 0.5 = 1.5 → conf = 1.0
        assert conf == pytest.approx(1.0)
        assert "toolbar" in templates

    def test_template_miss_does_not_add(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Notepad"],
            required_templates=["missing_tmpl"],
            match_threshold=0.0,
        )
        matcher = _mock_matcher({"missing_tmpl": False})
        az = ScreenStateAnalyzer(
            ocr_engine=_mock_ocr("Notepad"), template_matcher=matcher
        )
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, _, templates = az._score(fp, img, texts)
        # raw = 1, max = 1 + 0.5 = 1.5 → conf ≈ 0.667
        assert conf == pytest.approx(1.0 / 1.5, rel=1e-3)
        assert templates == []

    def test_empty_fingerprint_all_present_gives_1(self):
        fp = ScreenFingerprint(screen_id="empty", match_threshold=0.0)
        az = ScreenStateAnalyzer()
        img = _blank_image()
        conf, _, _ = az._score(fp, img, [])
        assert conf == pytest.approx(1.0)

    def test_negative_raw_clamped_to_zero(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["A", "B"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr())  # no text returned
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, _, _ = az._score(fp, img, texts)
        assert conf == pytest.approx(0.0)

    def test_confidence_clamped_to_one(self):
        """Confidence must never exceed 1.0 even with many optional hits."""
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["A"],
            optional_texts=["B", "C", "D", "E"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("A", "B", "C", "D", "E"))
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, _, _ = az._score(fp, img, texts)
        assert conf <= 1.0

    def test_template_absent_without_matcher_reduces_confidence(self):
        """When template_matcher is None, required_templates still count in max_score."""
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["A"],
            required_templates=["tmpl"],
            match_threshold=0.0,
        )
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("A"), template_matcher=None)
        img = _blank_image()
        ocr = az._run_ocr(img)
        texts = az._collect_texts(ocr)
        conf, _, _ = az._score(fp, img, texts)
        # raw=1.0, max=1.5 → conf≈0.667 (template skipped but still in denominator)
        assert conf == pytest.approx(1.0 / 1.5, rel=1e-3)


# ── ScreenStateAnalyzer — identify ────────────────────────────────────────────


class TestIdentify:
    def test_identifies_correct_screen(self):
        fp_notepad = ScreenFingerprint(
            screen_id="notepad_main",
            required_texts=["Notepad"],
            optional_texts=["File", "Edit"],
            match_threshold=0.6,
        )
        fp_dialog = ScreenFingerprint(
            screen_id="save_as_dialog",
            required_texts=["Save As"],
            match_threshold=0.6,
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp_notepad, fp_dialog],
            ocr_engine=_mock_ocr("Notepad", "File"),
        )
        result = az.identify(_blank_image())
        assert result.screen_id == "notepad_main"
        assert result.confidence > 0.6
        assert "Notepad" in result.matched_texts

    def test_returns_none_when_below_threshold(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=["Expected"],
            match_threshold=0.9,
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp],
            ocr_engine=_mock_ocr("Unrelated"),
        )
        result = az.identify(_blank_image())
        assert result.screen_id is None
        assert result.confidence == pytest.approx(0.0)

    def test_picks_highest_scoring_fingerprint(self):
        fp1 = ScreenFingerprint(
            screen_id="screen_a",
            required_texts=["Alpha"],
            match_threshold=0.5,
        )
        fp2 = ScreenFingerprint(
            screen_id="screen_b",
            required_texts=["Beta", "Gamma"],
            optional_texts=["Delta"],
            match_threshold=0.5,
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp1, fp2],
            ocr_engine=_mock_ocr("Beta", "Gamma", "Delta"),
        )
        result = az.identify(_blank_image())
        assert result.screen_id == "screen_b"

    def test_ocr_results_included_in_identification(self):
        fp = ScreenFingerprint(
            screen_id="s", required_texts=["X"], match_threshold=0.5
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp],
            ocr_engine=_mock_ocr("X"),
        )
        result = az.identify(_blank_image())
        assert len(result.ocr_results) > 0

    def test_no_fingerprints_returns_none(self):
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("Something"))
        result = az.identify(_blank_image())
        assert result.screen_id is None

    def test_no_ocr_engine(self):
        fp = ScreenFingerprint(
            screen_id="s", required_texts=[], match_threshold=0.0
        )
        az = ScreenStateAnalyzer(fingerprints=[fp])
        result = az.identify(_blank_image())
        assert result.screen_id == "s"


# ── ScreenStateAnalyzer — verify ──────────────────────────────────────────────


class TestVerify:
    def test_verify_correct_screen_returns_true(self):
        fp = ScreenFingerprint(
            screen_id="notepad_main",
            required_texts=["Notepad"],
            match_threshold=0.6,
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp],
            ocr_engine=_mock_ocr("Notepad"),
        )
        assert az.verify(_blank_image(), "notepad_main") is True

    def test_verify_wrong_screen_returns_false(self):
        fp = ScreenFingerprint(
            screen_id="notepad_main",
            required_texts=["Notepad"],
            match_threshold=0.6,
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp],
            ocr_engine=_mock_ocr("Notepad"),
        )
        assert az.verify(_blank_image(), "save_as_dialog") is False

    def test_verify_no_match_returns_false(self):
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr())
        assert az.verify(_blank_image(), "anything") is False


# ── ScreenStateAnalyzer — detect_popup ───────────────────────────────────────


class TestDetectPopup:
    def test_detects_error_popup(self):
        # Provide enough optional matches so error_popup scores above its threshold (0.5)
        # error_popup: required=["Error"], optional=["OK","Close","Cancel","failed","cannot","unable"]
        # With Error+OK+Close+Cancel: raw=1+0.3+0.3+0.3=1.9, max=1+1.8=2.8, conf≈0.68 > 0.5
        az = ScreenStateAnalyzer(ocr_engine=_mock_ocr("Error", "OK", "Close", "Cancel"))
        result = az.detect_popup(_blank_image())
        assert result is not None
        assert result.popup_type == "error"
        assert result.confidence > 0

    def test_detects_confirm_popup(self):
        # confirm_popup requires both "OK" and "Cancel" to avoid false positives
        az = ScreenStateAnalyzer(
            ocr_engine=_mock_ocr("OK", "Cancel", "Are you sure")
        )
        result = az.detect_popup(_blank_image())
        assert result is not None
        assert result.popup_type == "confirm"

    def test_detects_loading_popup(self):
        az = ScreenStateAnalyzer(
            ocr_engine=_mock_ocr("Please wait", "Loading")
        )
        result = az.detect_popup(_blank_image())
        assert result is not None
        assert result.popup_type == "loading"

    def test_no_popup_returns_none(self):
        az = ScreenStateAnalyzer(
            ocr_engine=_mock_ocr("Notepad", "File", "Edit")
        )
        result = az.detect_popup(_blank_image())
        assert result is None

    def test_popup_reuses_ocr_results(self):
        """detect_popup should not call OCR again when results are provided."""
        ocr_engine = _mock_ocr("Error", "OK")
        az = ScreenStateAnalyzer(ocr_engine=ocr_engine)
        pre_computed = [FakeOCRResult("Error", 0.9), FakeOCRResult("OK", 0.9)]
        az.detect_popup(_blank_image(), ocr_results=pre_computed)
        # recognize() should NOT have been called since results were provided
        ocr_engine.recognize.assert_not_called()

    def test_is_popup_flag_set_in_identify(self):
        fp = ScreenFingerprint(
            screen_id="notepad_main",
            required_texts=["Notepad"],
            match_threshold=0.5,
        )
        # Provide enough error-popup optional matches to exceed its threshold (0.5)
        az = ScreenStateAnalyzer(
            fingerprints=[fp],
            ocr_engine=_mock_ocr("Notepad", "Error", "OK", "Close"),
        )
        result = az.identify(_blank_image())
        assert result.is_popup is True

    def test_is_popup_false_when_no_popup(self):
        fp = ScreenFingerprint(
            screen_id="notepad_main",
            required_texts=["Notepad"],
            match_threshold=0.5,
        )
        az = ScreenStateAnalyzer(
            fingerprints=[fp],
            ocr_engine=_mock_ocr("Notepad", "File"),
        )
        result = az.identify(_blank_image())
        assert result.is_popup is False


# ── Text collection and fuzzy matching ───────────────────────────────────────


class TestCollectTexts:
    def test_collects_words(self):
        az = ScreenStateAnalyzer()
        results = [FakeOCRResult("Hello", 0.9), FakeOCRResult("World", 0.8)]
        texts = az._collect_texts(results)
        assert "Hello" in texts
        assert "World" in texts

    def test_includes_full_concatenation(self):
        az = ScreenStateAnalyzer()
        results = [FakeOCRResult("Save", 0.9), FakeOCRResult("As", 0.8)]
        texts = az._collect_texts(results)
        assert "Save As" in texts

    def test_empty_results_returns_empty(self):
        az = ScreenStateAnalyzer()
        assert az._collect_texts([]) == []

    def test_skips_blank_words(self):
        az = ScreenStateAnalyzer()
        results = [FakeOCRResult("  ", 0.9), FakeOCRResult("Hello", 0.8)]
        texts = az._collect_texts(results)
        assert "  " not in texts
        assert "Hello" in texts

    def test_calls_group_into_lines_when_available(self):
        ocr_engine = MagicMock()
        ocr_engine._group_into_lines.return_value = [
            FakeOCRResult("Save As", 0.9)
        ]
        az = ScreenStateAnalyzer(ocr_engine=ocr_engine)
        results = [FakeOCRResult("Save", 0.9), FakeOCRResult("As", 0.8)]
        texts = az._collect_texts(results)
        assert "Save As" in texts


class TestTextFound:
    def test_exact_substring_match(self):
        az = ScreenStateAnalyzer()
        assert az._text_found("Notepad", ["Untitled - Notepad"]) is True

    def test_case_insensitive(self):
        az = ScreenStateAnalyzer()
        assert az._text_found("notepad", ["Notepad"]) is True

    def test_not_found(self):
        az = ScreenStateAnalyzer()
        assert az._text_found("Login", ["Notepad", "File"]) is False

    def test_fuzzy_partial_match(self):
        az = ScreenStateAnalyzer(fuzzy_threshold=70)
        # "Save" appears as a substring of "Save As"
        assert az._text_found("Save", ["Save As"]) is True


# ── ScreenStateDetector (legacy API) ─────────────────────────────────────────


class TestScreenStateDetector:
    def test_register_and_identify(self):
        detector = ScreenStateDetector(ocr_engine=_mock_ocr("Sign In"))
        detector.register("login_screen", ocr_anchors=["Sign In"], threshold=0.5)
        result = detector.identify(_blank_image())
        assert result == "login_screen"

    def test_identify_returns_none_when_no_match(self):
        detector = ScreenStateDetector(ocr_engine=_mock_ocr("Unrelated"))
        detector.register("login_screen", ocr_anchors=["Sign In"], threshold=0.9)
        assert detector.identify(_blank_image()) is None

    def test_load_from_config(self):
        config = {
            "notepad_main": {
                "required_texts": ["Notepad"],
                "optional_texts": ["File"],
                "match_threshold": 0.5,
            },
            "save_as_dialog": {
                "required_texts": ["Save As"],
                "match_threshold": 0.5,
            },
        }
        detector = ScreenStateDetector(ocr_engine=_mock_ocr("Notepad", "File"))
        detector.load_from_config(config)
        result = detector.identify(_blank_image())
        assert result == "notepad_main"

    def test_load_from_config_defaults(self):
        """Minimal config dict should use ScreenFingerprint defaults."""
        config = {"simple": {}}
        detector = ScreenStateDetector()
        detector.load_from_config(config)
        # Empty fingerprint — should match anything (no required texts)
        result = detector.identify(_blank_image())
        assert result == "simple"

    def test_multiple_register_calls(self):
        detector = ScreenStateDetector(ocr_engine=_mock_ocr("Save As", "File name"))
        detector.register("notepad_main", ocr_anchors=["Notepad"], threshold=0.5)
        detector.register(
            "save_as_dialog", ocr_anchors=["Save As", "File name"], threshold=0.5
        )
        result = detector.identify(_blank_image())
        assert result == "save_as_dialog"

    def test_register_with_template_anchors(self):
        matcher = _mock_matcher({"toolbar": True})
        detector = ScreenStateDetector(
            ocr_engine=_mock_ocr(), template_matcher=matcher
        )
        detector.register(
            "main_screen",
            template_anchors=["toolbar"],
            threshold=0.3,
        )
        result = detector.identify(_blank_image())
        assert result == "main_screen"

    def test_identify_invalidates_analyzer_cache(self):
        """Registering a new state after first identify should still work."""
        detector = ScreenStateDetector(ocr_engine=_mock_ocr("Sign In"))
        # First identify with no states
        assert detector.identify(_blank_image()) is None
        # Register after the fact
        detector.register("login_screen", ocr_anchors=["Sign In"], threshold=0.5)
        result = detector.identify(_blank_image())
        assert result == "login_screen"

    def test_load_from_config_skips_invalid_entries(self):
        """Invalid fingerprint entries should be logged and skipped, not abort."""
        config = {
            "good_screen": {"required_texts": ["Notepad"], "match_threshold": 0.5},
            "bad_screen": {"match_threshold": 99.0},  # threshold out of range → Pydantic error
        }
        detector = ScreenStateDetector(ocr_engine=_mock_ocr("Notepad"))
        detector.load_from_config(config)
        # Only the good fingerprint should have been registered
        assert len(detector._fingerprints) == 1
        assert detector._fingerprints[0].screen_id == "good_screen"

    def test_fuzzy_threshold_propagated_to_analyzer(self):
        detector = ScreenStateDetector(
            ocr_engine=_mock_ocr("Sign In"), fuzzy_threshold=85
        )
        detector.register("login", ocr_anchors=["Sign In"], threshold=0.5)
        analyzer = detector._get_analyzer()
        assert analyzer._fuzzy_threshold == 85


# ── ScreenIdentification dataclass ───────────────────────────────────────────


class TestScreenIdentification:
    def test_default_fields(self):
        si = ScreenIdentification(screen_id="test", confidence=0.8)
        assert si.screen_id == "test"
        assert si.confidence == 0.8
        assert si.matched_texts == []
        assert si.matched_templates == []
        assert si.ocr_results == []
        assert si.is_popup is False

    def test_immutable(self):
        si = ScreenIdentification(screen_id="test", confidence=0.5)
        with pytest.raises((AttributeError, TypeError)):
            si.screen_id = "other"  # type: ignore[misc]


# ── PopupInfo dataclass ───────────────────────────────────────────────────────


class TestPopupInfo:
    def test_fields(self):
        pi = PopupInfo(popup_type="error", confidence=0.8, matched_texts=["Error"])
        assert pi.popup_type == "error"
        assert pi.confidence == 0.8
        assert pi.matched_texts == ["Error"]

    def test_immutable(self):
        pi = PopupInfo(popup_type="confirm", confidence=0.7)
        with pytest.raises((AttributeError, TypeError)):
            pi.popup_type = "error"  # type: ignore[misc]


# ── OCR error resilience ──────────────────────────────────────────────────────


class TestOCRResilience:
    def test_ocr_exception_returns_empty(self):
        engine = MagicMock()
        engine.recognize.side_effect = RuntimeError("OCR crashed")
        az = ScreenStateAnalyzer(ocr_engine=engine)
        results = az._run_ocr(_blank_image())
        assert results == []

    def test_identify_works_when_ocr_fails(self):
        fp = ScreenFingerprint(
            screen_id="s", required_texts=[], match_threshold=0.0
        )
        engine = MagicMock()
        engine.recognize.side_effect = RuntimeError("crash")
        az = ScreenStateAnalyzer(fingerprints=[fp], ocr_engine=engine)
        result = az.identify(_blank_image())
        assert result.screen_id == "s"

    def test_template_exception_is_caught(self):
        fp = ScreenFingerprint(
            screen_id="s",
            required_texts=[],
            required_templates=["bad_tmpl"],
            match_threshold=0.0,
        )
        matcher = MagicMock()
        matcher.find_one.side_effect = RuntimeError("template crash")
        az = ScreenStateAnalyzer(fingerprints=[fp], template_matcher=matcher)
        result = az.identify(_blank_image())
        # Should not raise, just miss the template
        assert result is not None
