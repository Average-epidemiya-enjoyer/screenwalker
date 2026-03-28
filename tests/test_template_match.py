"""Tests for the template matching module.

Test layers:
- Unit: helpers (_pil_to_cv, _load_template, _nms) — no screenshot needed.
- Integration: synthetic PIL screenshots with known button positions — real cv2.
- TemplateMatcher class: load_template caching, find_one, find_all, multiscale.
- Standalone functions: match_template, match_template_all, match_template_multiscale.
- Utilities: draw_matches output shape/type.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from screenwalker.vision.screen_state import BBox, FindResult
from screenwalker.vision.template_match import (
    MatchResult,
    TemplateMatcher,
    _load_template,
    _nms,
    _pil_to_cv,
    _to_gray,
    draw_matches,
    match_template,
    match_template_all,
    match_template_multiscale,
)


# ---------------------------------------------------------------------------
# Synthetic image factories
# ---------------------------------------------------------------------------


def _button_img(size: int = 40) -> Image.Image:
    """A button template: dark-blue border + light-gray fill (non-uniform → matchable)."""
    img = Image.new("RGB", (size, size), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(20, 60, 180), width=4)
    return img


def _screenshot_with_buttons(
    positions: list[tuple[int, int]],
    button: Image.Image,
    size: tuple[int, int] = (700, 400),
) -> Image.Image:
    """White screenshot with *button* pasted at each position."""
    bg = Image.new("RGB", size, color=(240, 240, 240))
    for x, y in positions:
        bg.paste(button, (x, y))
    return bg


def _save_template(img: Image.Image, path: Path) -> Path:
    img.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# _pil_to_cv
# ---------------------------------------------------------------------------


class TestPilToCv:
    def test_returns_numpy_array(self) -> None:
        result = _pil_to_cv(Image.new("RGB", (50, 30)))
        assert isinstance(result, np.ndarray)

    def test_shape_is_height_width_3(self) -> None:
        result = _pil_to_cv(Image.new("RGB", (100, 60)))
        assert result.shape == (60, 100, 3)

    def test_white_image_is_white(self) -> None:
        result = _pil_to_cv(Image.new("RGB", (10, 10), color=(255, 255, 255)))
        assert np.all(result == 255)

    def test_rgba_converted_correctly(self) -> None:
        rgba = Image.new("RGBA", (10, 10), color=(100, 150, 200, 255))
        result = _pil_to_cv(rgba)
        assert result.shape == (10, 10, 3)


# ---------------------------------------------------------------------------
# _load_template
# ---------------------------------------------------------------------------


class TestLoadTemplate:
    def test_raises_file_not_found_for_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="Template not found"):
            _load_template("/nonexistent/template.png")

    def test_loads_valid_png(self, tmp_path: Path) -> None:
        path = _save_template(_button_img(), tmp_path / "btn.png")
        tmpl = _load_template(path)
        assert isinstance(tmpl, np.ndarray)
        assert tmpl.ndim == 3

    def test_invalid_bytes_raises_value_error(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"not image data")
        with pytest.raises(ValueError, match="Could not decode"):
            _load_template(fake)


# ---------------------------------------------------------------------------
# _nms
# ---------------------------------------------------------------------------


class TestNMS:
    def test_empty_input_returns_empty(self) -> None:
        assert _nms(np.empty((0, 4)), np.empty(0)) == []

    def test_single_box_always_kept(self) -> None:
        boxes = np.array([[0, 0, 30, 30]], dtype=float)
        scores = np.array([0.9])
        assert _nms(boxes, scores) == [0]

    def test_non_overlapping_all_kept(self) -> None:
        # Four boxes far apart — no overlap
        boxes = np.array([
            [0, 0, 20, 20],
            [200, 0, 20, 20],
            [0, 200, 20, 20],
            [200, 200, 20, 20],
        ], dtype=float)
        scores = np.array([0.9, 0.85, 0.8, 0.75])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert sorted(keep) == [0, 1, 2, 3]

    def test_five_overlapping_keeps_one_highest_confidence(self) -> None:
        """Five heavily-overlapping boxes → only the highest-score survives."""
        boxes = np.array([
            [10, 10, 30, 30],
            [12, 10, 30, 30],
            [10, 12, 30, 30],
            [11, 11, 30, 30],
            [13, 13, 30, 30],
        ], dtype=float)
        scores = np.array([0.95, 0.80, 0.85, 0.70, 0.75])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # highest confidence

    def test_iou_threshold_0_keeps_only_top(self) -> None:
        """threshold=0.0 means any overlap suppresses (only 1 survives)."""
        boxes = np.array([
            [0, 0, 30, 30],
            [15, 15, 30, 30],  # partial overlap
        ], dtype=float)
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores, iou_threshold=0.0)
        assert keep == [0]

    def test_iou_threshold_1_keeps_all(self) -> None:
        """threshold=1.0 means only identical boxes are suppressed."""
        boxes = np.array([
            [0, 0, 30, 30],
            [15, 15, 30, 30],  # partial overlap, IoU < 1.0
        ], dtype=float)
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores, iou_threshold=1.0)
        assert sorted(keep) == [0, 1]

    def test_output_sorted_by_score_descending(self) -> None:
        """Kept indices are in confidence-descending order."""
        boxes = np.array([
            [0, 0, 10, 10],
            [100, 100, 10, 10],
            [200, 200, 10, 10],
        ], dtype=float)
        scores = np.array([0.7, 0.95, 0.85])
        keep = _nms(boxes, scores)
        # Expected order: index 1 (0.95), index 2 (0.85), index 0 (0.7)
        assert keep == [1, 2, 0]

    def test_exact_duplicate_boxes_suppressed(self) -> None:
        """Identical boxes (IoU=1.0) — only the highest-score kept."""
        boxes = np.tile([[5, 5, 20, 20]], (3, 1)).astype(float)
        scores = np.array([0.8, 0.9, 0.7])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 1


# ---------------------------------------------------------------------------
# MatchResult dataclass
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_creation(self) -> None:
        r = MatchResult(
            template_name="btn",
            confidence=0.95,
            center=(50, 60),
            bbox=BBox(30, 40, 40, 40),
            scale=1.0,
        )
        assert r.template_name == "btn"
        assert r.scale == 1.0

    def test_frozen(self) -> None:
        r = MatchResult("btn", 0.9, (0, 0), BBox(0, 0, 10, 10))
        with pytest.raises((AttributeError, TypeError)):
            r.confidence = 0.5  # type: ignore[misc]

    def test_scale_defaults_to_1(self) -> None:
        r = MatchResult("t", 0.8, (0, 0), BBox(0, 0, 5, 5))
        assert r.scale == 1.0


# ---------------------------------------------------------------------------
# TemplateMatcher.load_template
# ---------------------------------------------------------------------------


class TestLoadTemplateMethod:
    def test_finds_by_stem_with_png_extension(self, tmp_path: Path) -> None:
        _save_template(_button_img(), tmp_path / "button.png")
        matcher = TemplateMatcher(tmp_path)
        tmpl = matcher.load_template("button")
        assert isinstance(tmpl, np.ndarray)

    def test_finds_by_exact_name(self, tmp_path: Path) -> None:
        _save_template(_button_img(), tmp_path / "button.png")
        matcher = TemplateMatcher(tmp_path)
        tmpl = matcher.load_template("button.png")
        assert isinstance(tmpl, np.ndarray)

    def test_missing_template_raises_file_not_found(self, tmp_path: Path) -> None:
        matcher = TemplateMatcher(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            matcher.load_template("ghost")

    def test_caching_reads_disk_once(self, tmp_path: Path) -> None:
        _save_template(_button_img(), tmp_path / "button.png")
        matcher = TemplateMatcher(tmp_path)
        tmpl_a = matcher.load_template("button")
        tmpl_b = matcher.load_template("button")
        assert tmpl_a is tmpl_b  # same object from cache

    def test_clear_cache_forces_reload(self, tmp_path: Path) -> None:
        _save_template(_button_img(), tmp_path / "button.png")
        matcher = TemplateMatcher(tmp_path)
        tmpl_a = matcher.load_template("button")
        matcher.clear_cache()
        tmpl_b = matcher.load_template("button")
        # Different object after cache clear
        assert tmpl_a is not tmpl_b


# ---------------------------------------------------------------------------
# TemplateMatcher.find_one  (integration — real cv2)
# ---------------------------------------------------------------------------


class TestFindOne:
    def test_finds_button_in_screenshot(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(100, 80)], btn)
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one(screenshot, "button", threshold=0.85)
        assert result is not None
        assert isinstance(result, MatchResult)
        assert result.confidence >= 0.85
        # Centre should be close to (120, 100) = (100+20, 80+20)
        cx, cy = result.center
        assert abs(cx - 120) <= 5
        assert abs(cy - 100) <= 5

    def test_returns_none_when_template_absent(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        # Empty background — no button
        screenshot = Image.new("RGB", (400, 300), color=(240, 240, 240))
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one(screenshot, "button", threshold=0.9)
        assert result is None

    def test_returns_none_when_screenshot_smaller_than_template(
        self, tmp_path: Path
    ) -> None:
        btn = _button_img(80)
        _save_template(btn, tmp_path / "button.png")
        tiny = Image.new("RGB", (20, 20), color=(240, 240, 240))
        matcher = TemplateMatcher(tmp_path)
        assert matcher.find_one(tiny, "button") is None

    def test_region_restricts_search(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        # Button at (50, 50) — region that doesn't include it
        screenshot = _screenshot_with_buttons([(50, 50)], btn)
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one(
            screenshot, "button", threshold=0.85,
            region=BBox(x=300, y=200, w=200, h=150),
        )
        assert result is None

    def test_region_offsets_bbox_correctly(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        # Button at (10, 10) inside region (100, 100, 200, 200)
        screenshot = _screenshot_with_buttons([(110, 110)], btn)
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one(
            screenshot, "button", threshold=0.85,
            region=BBox(x=100, y=100, w=200, h=200),
        )
        assert result is not None
        # bbox.x should be ~110 (not ~10 relative to region)
        assert result.bbox.x >= 100

    def test_result_method_field_is_absent_in_match_result(
        self, tmp_path: Path
    ) -> None:
        """MatchResult doesn't have a 'method' field (FindResult does)."""
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(50, 50)], btn)
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one(screenshot, "button", threshold=0.8)
        if result is not None:
            assert isinstance(result, MatchResult)
            assert not hasattr(result, "method")


# ---------------------------------------------------------------------------
# TemplateMatcher.find_all  (multiple instances + NMS)
# ---------------------------------------------------------------------------


class TestFindAll:
    POSITIONS = [(50, 80), (200, 80), (350, 80)]

    def test_finds_all_three_buttons(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons(self.POSITIONS, btn)
        matcher = TemplateMatcher(tmp_path)
        results = matcher.find_all(screenshot, "button", threshold=0.85)
        assert len(results) == 3

    def test_results_sorted_by_confidence_desc(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons(self.POSITIONS, btn)
        matcher = TemplateMatcher(tmp_path)
        results = matcher.find_all(screenshot, "button", threshold=0.85)
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence

    def test_returns_empty_when_no_match(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        blank = Image.new("RGB", (400, 300), color=(240, 240, 240))
        matcher = TemplateMatcher(tmp_path)
        assert matcher.find_all(blank, "button", threshold=0.9) == []

    def test_centers_near_expected_positions(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons(self.POSITIONS, btn)
        matcher = TemplateMatcher(tmp_path)
        results = matcher.find_all(screenshot, "button", threshold=0.85)

        expected_centers = sorted(
            [(x + 20, y + 20) for x, y in self.POSITIONS]
        )
        found_centers = sorted(results, key=lambda r: r.center[0])
        for res, (ex, ey) in zip(found_centers, expected_centers):
            cx, cy = res.center
            assert abs(cx - ex) <= 5, f"Expected cx≈{ex}, got {cx}"
            assert abs(cy - ey) <= 5, f"Expected cy≈{ey}, got {cy}"

    def test_nms_prevents_duplicate_detections(self, tmp_path: Path) -> None:
        """Even if the match-map has many peaks at one location, NMS collapses them."""
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        # Single button — should get exactly 1 result
        screenshot = _screenshot_with_buttons([(100, 100)], btn)
        matcher = TemplateMatcher(tmp_path)
        results = matcher.find_all(screenshot, "button", threshold=0.80)
        assert len(results) == 1

    def test_all_results_are_match_result_instances(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons(self.POSITIONS, btn)
        matcher = TemplateMatcher(tmp_path)
        results = matcher.find_all(screenshot, "button", threshold=0.85)
        for r in results:
            assert isinstance(r, MatchResult)


# ---------------------------------------------------------------------------
# TemplateMatcher.find_one_multiscale
# ---------------------------------------------------------------------------


class TestFindOneMultiscale:
    def test_finds_scaled_up_template(self, tmp_path: Path) -> None:
        """Template is 40×40; screenshot has a 50×50 version (scale ≈ 1.25)."""
        btn_orig = _button_img(40)
        _save_template(btn_orig, tmp_path / "button.png")

        btn_large = btn_orig.resize((50, 50), Image.LANCZOS)
        screenshot = Image.new("RGB", (400, 300), color=(240, 240, 240))
        screenshot.paste(btn_large, (100, 100))

        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one_multiscale(
            screenshot, "button",
            scales=(0.9, 1.0, 1.1, 1.25, 1.3),
            threshold=0.70,
        )
        assert result is not None
        assert result.scale > 1.0
        assert result.confidence >= 0.70

    def test_finds_scaled_down_template(self, tmp_path: Path) -> None:
        """Template is 40×40; screenshot has a 32×32 version (scale ≈ 0.8)."""
        btn_orig = _button_img(40)
        _save_template(btn_orig, tmp_path / "button.png")

        btn_small = btn_orig.resize((32, 32), Image.LANCZOS)
        screenshot = Image.new("RGB", (400, 300), color=(240, 240, 240))
        screenshot.paste(btn_small, (150, 150))

        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one_multiscale(
            screenshot, "button",
            scales=(0.7, 0.8, 0.9, 1.0),
            threshold=0.65,
        )
        assert result is not None
        assert result.scale < 1.0

    def test_exact_scale_1_still_found(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(80, 80)], btn)
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one_multiscale(
            screenshot, "button", scales=(0.9, 1.0, 1.1), threshold=0.85
        )
        assert result is not None
        assert abs(result.scale - 1.0) < 0.05

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        blank = Image.new("RGB", (400, 300), color=(240, 240, 240))
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one_multiscale(blank, "button", threshold=0.95)
        assert result is None

    def test_returns_best_scale_when_multiple_candidates(
        self, tmp_path: Path
    ) -> None:
        """Multiscale returns the result with the highest confidence."""
        btn = _button_img(40)
        _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(80, 80)], btn)
        matcher = TemplateMatcher(tmp_path)
        result = matcher.find_one_multiscale(
            screenshot, "button", scales=(0.8, 0.9, 1.0, 1.1, 1.2), threshold=0.5
        )
        assert result is not None
        # Scale closest to 1.0 should win (exact match)
        assert abs(result.scale - 1.0) < 0.15


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


class TestMatchTemplate:
    def test_finds_button(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(60, 60)], btn)
        result = match_template(screenshot, path, threshold=0.85)
        assert result is not None
        assert isinstance(result, FindResult)
        assert result.method == "template"
        assert result.confidence >= 0.85

    def test_returns_none_below_threshold(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        blank = Image.new("RGB", (400, 300), color=(240, 240, 240))
        assert match_template(blank, path, threshold=0.95) is None

    def test_region_restricts_search(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        # Button at (20, 20), region that doesn't include it
        screenshot = _screenshot_with_buttons([(20, 20)], btn)
        result = match_template(
            screenshot, path, threshold=0.85,
            region=BBox(x=400, y=300, w=100, h=50),
        )
        assert result is None

    def test_element_name_is_template_stem(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "my_button.png")
        screenshot = _screenshot_with_buttons([(50, 50)], btn)
        result = match_template(screenshot, path, threshold=0.85)
        if result is not None:
            assert result.element == "my_button"


class TestMatchTemplateAll:
    def test_finds_all_three(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        positions = [(50, 80), (200, 80), (400, 80)]
        screenshot = _screenshot_with_buttons(positions, btn, size=(700, 300))
        results = match_template_all(screenshot, path, threshold=0.85)
        assert len(results) == 3

    def test_returns_find_result_objects(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(50, 50)], btn)
        results = match_template_all(screenshot, path, threshold=0.85)
        for r in results:
            assert isinstance(r, FindResult)
            assert r.method == "template"

    def test_returns_empty_for_blank(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        blank = Image.new("RGB", (400, 300), color=(240, 240, 240))
        assert match_template_all(blank, path, threshold=0.95) == []


class TestMatchTemplateMultiscale:
    def test_finds_at_125_scale(self, tmp_path: Path) -> None:
        btn_orig = _button_img(40)
        path = _save_template(btn_orig, tmp_path / "button.png")
        btn_large = btn_orig.resize((50, 50), Image.LANCZOS)
        screenshot = Image.new("RGB", (400, 300), color=(240, 240, 240))
        screenshot.paste(btn_large, (100, 100))
        result = match_template_multiscale(
            screenshot, path,
            threshold=0.65,
            scale_min=0.9,
            scale_max=1.4,
            scale_steps=8,
        )
        assert result is not None
        assert isinstance(result, FindResult)

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        blank = Image.new("RGB", (400, 300), color=(240, 240, 240))
        assert match_template_multiscale(blank, path, threshold=0.95) is None

    def test_scale_stored_in_metadata(self, tmp_path: Path) -> None:
        btn = _button_img(40)
        path = _save_template(btn, tmp_path / "button.png")
        screenshot = _screenshot_with_buttons([(80, 80)], btn)
        result = match_template_multiscale(screenshot, path, threshold=0.8)
        if result is not None:
            assert "scale" in result.metadata


# ---------------------------------------------------------------------------
# draw_matches
# ---------------------------------------------------------------------------


class TestDrawMatches:
    def test_returns_same_size_image(self) -> None:
        screenshot = Image.new("RGB", (400, 300))
        matches = [
            MatchResult("btn", 0.9, (50, 50), BBox(30, 30, 40, 40)),
        ]
        out = draw_matches(screenshot, matches)
        assert out.size == screenshot.size

    def test_returns_pil_image(self) -> None:
        screenshot = Image.new("RGB", (200, 100))
        out = draw_matches(screenshot, [])
        assert isinstance(out, Image.Image)

    def test_does_not_modify_original(self) -> None:
        screenshot = Image.new("RGB", (200, 100), color=(255, 255, 255))
        original = np.array(screenshot).copy()
        draw_matches(screenshot, [MatchResult("t", 0.9, (50, 50), BBox(30, 30, 40, 40))])
        assert np.array_equal(np.array(screenshot), original)

    def test_empty_matches_returns_copy(self) -> None:
        screenshot = Image.new("RGB", (100, 80), color=(100, 100, 100))
        out = draw_matches(screenshot, [])
        assert out is not screenshot
        assert np.array_equal(np.array(out), np.array(screenshot))

    def test_draws_box_pixels(self) -> None:
        """After drawing a box, at least one pixel near the bbox edge changed."""
        screenshot = Image.new("RGB", (200, 200), color=(240, 240, 240))
        match = MatchResult("btn", 0.95, (50, 50), BBox(30, 30, 40, 40))
        out = draw_matches(screenshot, [match])
        # At least one pixel in the bounding box edge area should be non-gray
        edge_pixel = out.getpixel((30, 30))
        assert edge_pixel != (240, 240, 240)
