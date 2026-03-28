"""Tests for screenwalker.vision.detector.

Unit tests use a mocked ultralytics YOLO model so they run without any
model weights.  The integration test at the bottom requires
``models/icon_detect/best.pt`` and is automatically skipped otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from screenwalker.vision.detector import (
    NullDetector,
    UIElementDetector,
    DetectionResult,
    create_detector,
    normalise_class,
    UI_CLASS_ALIASES,
)
from screenwalker.vision.screen_state import BBox, FindResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_image(w: int = 100, h: int = 80) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")


def _make_box(cls_idx: int, conf: float, x1: float, y1: float, x2: float, y2: float) -> Any:
    """Build a mock ultralytics box object."""
    box = MagicMock()
    box.cls.item.return_value = float(cls_idx)
    box.conf.item.return_value = conf
    box.xyxy = [[x1, y1, x2, y2]]
    return box


def _make_yolo_result(boxes: list[Any]) -> Any:
    result = MagicMock()
    result.boxes = boxes
    return result


def _mock_model(class_names: dict[int, str], boxes: list[Any]) -> MagicMock:
    model = MagicMock()
    model.names = class_names
    model.predict.return_value = [_make_yolo_result(boxes)]
    return model


# ---------------------------------------------------------------------------
# normalise_class
# ---------------------------------------------------------------------------


class TestNormaliseClass:
    def test_canonical_names_map_to_themselves(self) -> None:
        for canonical in UI_CLASS_ALIASES:
            assert normalise_class(canonical) == canonical

    def test_alias_maps_to_canonical(self) -> None:
        assert normalise_class("btn") == "button"
        assert normalise_class("input") == "text_field"
        assert normalise_class("edit") == "text_field"
        assert normalise_class("combo") == "dropdown"
        assert normalise_class("hyperlink") == "link"

    def test_case_insensitive(self) -> None:
        assert normalise_class("BTN") == "button"
        assert normalise_class("ICON") == "icon"

    def test_unknown_name_returned_lowercased(self) -> None:
        assert normalise_class("UnknownWidget") == "unknownwidget"
        assert normalise_class("custom_element") == "custom_element"

    def test_whitespace_stripped(self) -> None:
        assert normalise_class("  button  ") == "button"
        assert normalise_class("  btn  ") == "button"


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_center(self) -> None:
        bbox = BBox(10, 20, 100, 50)
        det = DetectionResult(class_name="button", confidence=0.9, bbox=bbox)
        assert det.center == (60, 45)

    def test_to_find_result(self) -> None:
        bbox = BBox(5, 5, 50, 30)
        det = DetectionResult(class_name="icon", confidence=0.75, bbox=bbox)
        fr = det.to_find_result()
        assert isinstance(fr, FindResult)
        assert fr.element == "icon"
        assert fr.confidence == 0.75
        assert fr.bbox == bbox
        assert fr.method == "yolo"
        assert fr.metadata == {"yolo_class": "icon"}

    def test_frozen(self) -> None:
        det = DetectionResult(class_name="button", confidence=0.8, bbox=BBox(0, 0, 10, 10))
        with pytest.raises((AttributeError, TypeError)):
            det.class_name = "icon"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UIElementDetector (with mocked YOLO)
# ---------------------------------------------------------------------------


class TestUIElementDetectorInit:
    def test_available_false_when_no_model_path(self) -> None:
        det = UIElementDetector()
        assert not det.available

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            UIElementDetector(model_path=tmp_path / "nonexistent.pt")

    def test_import_error_raised_when_ultralytics_missing(self, tmp_path: Path) -> None:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                UIElementDetector(model_path=pt)

    def test_model_loaded_sets_available(self, tmp_path: Path) -> None:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        mock_model = _mock_model({0: "button"}, [])
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = UIElementDetector(model_path=pt)
        assert det.available
        assert det.class_names == {0: "button"}


class TestUIElementDetectorDetect:
    @pytest.fixture()
    def detector(self, tmp_path: Path) -> UIElementDetector:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        boxes = [
            _make_box(0, 0.95, 10.0, 20.0, 110.0, 70.0),
            _make_box(1, 0.70, 50.0, 50.0, 200.0, 100.0),
        ]
        mock_model = _mock_model({0: "button", 1: "input"}, boxes)
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = UIElementDetector(model_path=pt, confidence_threshold=0.5)
        det._model = mock_model  # inject so predict works
        return det

    def test_detect_returns_sorted_by_confidence(self, detector: UIElementDetector) -> None:
        image = _rgb_image()
        results = detector.detect(image)
        assert len(results) == 2
        assert results[0].confidence >= results[1].confidence

    def test_detect_normalises_class_names(self, detector: UIElementDetector) -> None:
        image = _rgb_image()
        results = detector.detect(image)
        classes = {r.class_name for r in results}
        assert "button" in classes
        assert "text_field" in classes  # "input" → "text_field"

    def test_detect_bbox_coordinates(self, detector: UIElementDetector) -> None:
        image = _rgb_image()
        results = detector.detect(image)
        button = next(r for r in results if r.class_name == "button")
        assert button.bbox == BBox(10, 20, 100, 50)

    def test_detect_with_region_offsets(self, detector: UIElementDetector) -> None:
        image = _rgb_image(400, 400)
        region = BBox(100, 200, 300, 200)
        results = detector.detect(image, region=region)
        button = next(r for r in results if r.class_name == "button")
        # bbox coords should be offset by region origin
        assert button.bbox.x == 10 + 100
        assert button.bbox.y == 20 + 200

    def test_detect_returns_empty_when_not_available(self) -> None:
        det = UIElementDetector()
        assert det.detect(_rgb_image()) == []

    def test_detect_returns_empty_on_predict_exception(self, detector: UIElementDetector) -> None:
        detector._model.predict.side_effect = RuntimeError("GPU OOM")
        assert detector.detect(_rgb_image()) == []


class TestFindByClass:
    @pytest.fixture()
    def detector(self, tmp_path: Path) -> UIElementDetector:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        boxes = [
            _make_box(0, 0.90, 0.0, 0.0, 50.0, 30.0),
            _make_box(0, 0.80, 60.0, 0.0, 110.0, 30.0),
            _make_box(1, 0.85, 0.0, 40.0, 100.0, 70.0),
        ]
        mock_model = _mock_model({0: "button", 1: "input"}, boxes)
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = UIElementDetector(model_path=pt)
        det._model = mock_model
        return det

    def test_find_by_class_filters_correctly(self, detector: UIElementDetector) -> None:
        image = _rgb_image(200, 100)
        buttons = detector.find_by_class(image, "button")
        assert len(buttons) == 2
        assert all(r.class_name == "button" for r in buttons)

    def test_find_by_class_alias(self, detector: UIElementDetector) -> None:
        image = _rgb_image(200, 100)
        fields = detector.find_by_class(image, "input")
        assert len(fields) == 1
        assert fields[0].class_name == "text_field"

    def test_find_by_class_empty_when_not_found(self, detector: UIElementDetector) -> None:
        image = _rgb_image(200, 100)
        assert detector.find_by_class(image, "checkbox") == []


class TestFindNearest:
    @pytest.fixture()
    def detector(self, tmp_path: Path) -> UIElementDetector:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        boxes = [
            _make_box(0, 0.9, 0.0, 0.0, 20.0, 20.0),    # button near (10,10)
            _make_box(0, 0.85, 80.0, 80.0, 100.0, 100.0),  # button far at (90,90)
        ]
        mock_model = _mock_model({0: "button"}, boxes)
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = UIElementDetector(model_path=pt)
        det._model = mock_model
        return det

    def test_returns_nearest_to_anchor(self, detector: UIElementDetector) -> None:
        image = _rgb_image(200, 200)
        # OCR engine mock: anchor text found at (10, 10)
        ocr = MagicMock()
        anchor_bbox = BBox(0, 0, 20, 20)
        ocr.find_text.return_value = FindResult(
            element="Delete", confidence=1.0, bbox=anchor_bbox, method="ocr"
        )
        result = detector.find_nearest(image, "button", "Delete", ocr)
        assert result is not None
        # nearest button center is (10, 10) vs far button at (90, 90)
        assert result.bbox.center == (10, 10)

    def test_returns_none_when_anchor_not_found(self, detector: UIElementDetector) -> None:
        image = _rgb_image()
        ocr = MagicMock()
        ocr.find_text.return_value = None
        assert detector.find_nearest(image, "button", "Confirm", ocr) is None

    def test_returns_none_when_no_elements(self, tmp_path: Path) -> None:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        mock_model = _mock_model({}, [])
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = UIElementDetector(model_path=pt)
        det._model = mock_model
        ocr = MagicMock()
        ocr.find_text.return_value = FindResult(
            element="OK", confidence=1.0, bbox=BBox(0, 0, 20, 20), method="ocr"
        )
        assert det.find_nearest(_rgb_image(), "button", "OK", ocr) is None


class TestFinderProtocol:
    @pytest.fixture()
    def detector(self, tmp_path: Path) -> UIElementDetector:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        boxes = [
            _make_box(0, 0.9, 10.0, 10.0, 60.0, 40.0),
            _make_box(0, 0.6, 70.0, 10.0, 120.0, 40.0),
        ]
        mock_model = _mock_model({0: "button"}, boxes)
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            det = UIElementDetector(model_path=pt, confidence_threshold=0.5)
        det._model = mock_model
        return det

    def test_find_returns_best_result(self, detector: UIElementDetector) -> None:
        result = detector.find(_rgb_image(), "button")
        assert result is not None
        assert result.confidence == 0.9

    def test_find_returns_none_below_threshold(self, detector: UIElementDetector) -> None:
        result = detector.find(_rgb_image(), "button", threshold=0.95)
        assert result is None

    def test_find_returns_none_for_unknown_class(self, detector: UIElementDetector) -> None:
        result = detector.find(_rgb_image(), "slider")
        assert result is None

    def test_find_all_returns_all_above_threshold(self, detector: UIElementDetector) -> None:
        results = detector.find_all(_rgb_image(), "button", threshold=0.5)
        assert len(results) == 2
        assert all(r.confidence >= 0.5 for r in results)

    def test_find_all_filters_by_threshold(self, detector: UIElementDetector) -> None:
        results = detector.find_all(_rgb_image(), "button", threshold=0.8)
        assert len(results) == 1
        assert results[0].confidence == 0.9

    def test_detect_all_returns_all(self, detector: UIElementDetector) -> None:
        results = detector.detect_all(_rgb_image())
        assert len(results) == 2


# ---------------------------------------------------------------------------
# NullDetector
# ---------------------------------------------------------------------------


class TestNullDetector:
    def setup_method(self) -> None:
        self.det = NullDetector()

    def test_available_false(self) -> None:
        assert not self.det.available

    def test_detect_empty(self) -> None:
        assert self.det.detect(_rgb_image()) == []

    def test_find_by_class_empty(self) -> None:
        assert self.det.find_by_class(_rgb_image(), "button") == []

    def test_find_nearest_none(self) -> None:
        assert self.det.find_nearest(_rgb_image(), "button", "OK", MagicMock()) is None

    def test_find_none(self) -> None:
        assert self.det.find(_rgb_image(), "button") is None

    def test_find_all_empty(self) -> None:
        assert self.det.find_all(_rgb_image(), "button") == []

    def test_detect_all_empty(self) -> None:
        assert self.det.detect_all(_rgb_image()) == []


# ---------------------------------------------------------------------------
# create_detector factory
# ---------------------------------------------------------------------------


class TestCreateDetector:
    def _cfg(self, **kwargs: Any) -> SimpleNamespace:
        defaults = dict(yolo_enabled=False, yolo_model_path=None, yolo_confidence=0.5)
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_returns_null_when_disabled(self) -> None:
        assert isinstance(create_detector(self._cfg(yolo_enabled=False)), NullDetector)

    def test_returns_null_when_model_path_missing(self, tmp_path: Path) -> None:
        cfg = self._cfg(yolo_enabled=True, yolo_model_path=str(tmp_path / "nope.pt"))
        assert isinstance(create_detector(cfg), NullDetector)

    def test_returns_null_when_no_model_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ensure default path does not exist
        monkeypatch.chdir(tmp_path_for_cwd())
        cfg = self._cfg(yolo_enabled=True)
        assert isinstance(create_detector(cfg), NullDetector)

    def test_loads_explicit_model_path(self, tmp_path: Path) -> None:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        mock_model = _mock_model({0: "button"}, [])
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            cfg = self._cfg(yolo_enabled=True, yolo_model_path=str(pt))
            det = create_detector(cfg)
        assert isinstance(det, UIElementDetector)
        assert det.confidence_threshold == 0.5

    def test_import_error_falls_back_to_null(self, tmp_path: Path) -> None:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        with patch.dict("sys.modules", {"ultralytics": None}):
            cfg = self._cfg(yolo_enabled=True, yolo_model_path=str(pt))
            det = create_detector(cfg)
        assert isinstance(det, NullDetector)

    def test_respects_confidence_from_config(self, tmp_path: Path) -> None:
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"fake")
        mock_model = _mock_model({}, [])
        mock_yolo_cls = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            cfg = self._cfg(yolo_enabled=True, yolo_model_path=str(pt), yolo_confidence=0.75)
            det = create_detector(cfg)
        assert isinstance(det, UIElementDetector)
        assert det.confidence_threshold == 0.75


def tmp_path_for_cwd(subdir: str = "empty_cwd") -> Path:
    """Return a stable temp directory that has no model weights."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    return d


# ---------------------------------------------------------------------------
# Integration test — requires real model weights
# ---------------------------------------------------------------------------


_WEIGHTS_PATH = Path("models/icon_detect/best.pt")


@pytest.mark.skipif(
    not _WEIGHTS_PATH.exists(),
    reason="OmniParser weights not found — run: python scripts/download_model.py",
)
class TestIntegrationWithRealModel:
    @pytest.fixture(scope="class")
    def detector(self) -> UIElementDetector:
        return UIElementDetector(model_path=_WEIGHTS_PATH, confidence_threshold=0.3)

    def test_model_loads(self, detector: UIElementDetector) -> None:
        assert detector.available

    def test_detect_on_blank_image_does_not_crash(self, detector: UIElementDetector) -> None:
        image = _rgb_image(1920, 1080)
        results = detector.detect(image)
        assert isinstance(results, list)

    def test_find_returns_find_result_or_none(self, detector: UIElementDetector) -> None:
        image = _rgb_image(1920, 1080)
        result = detector.find(image, "button")
        assert result is None or isinstance(result, FindResult)
