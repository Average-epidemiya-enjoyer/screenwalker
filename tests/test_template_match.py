"""Tests for the template matching module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from screenwalker.vision.screen_state import BBox, FindResult
from screenwalker.vision.template_match import (
    _load_template,
    _pil_to_cv,
    match_template,
    match_template_multiscale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white_image(width: int = 300, height: int = 200) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _red_square_template(tmp_path: Path, size: int = 20) -> Path:
    """Write a small red-square PNG to tmp_path and return its path."""
    img = Image.new("RGB", (size, size), color=(255, 0, 0))
    p = tmp_path / "red_square.png"
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# _pil_to_cv
# ---------------------------------------------------------------------------


class TestPilToCv:
    """Tests for the PIL → OpenCV conversion helper."""

    def test_returns_numpy_array(self) -> None:
        """Output is a numpy ndarray."""
        result = _pil_to_cv(_white_image())
        assert isinstance(result, np.ndarray)

    def test_shape_matches_image(self) -> None:
        """Shape is (height, width, 3) for an RGB image."""
        img = _white_image(100, 60)
        result = _pil_to_cv(img)
        assert result.shape == (60, 100, 3)

    def test_white_image_is_white_in_bgr(self) -> None:
        """A white PIL image produces a white BGR array."""
        result = _pil_to_cv(_white_image(10, 10))
        # White in BGR = (255, 255, 255)
        assert np.all(result == 255)


# ---------------------------------------------------------------------------
# _load_template
# ---------------------------------------------------------------------------


class TestLoadTemplate:
    """Tests for the template loader helper."""

    def test_raises_file_not_found_for_missing_file(self) -> None:
        """Missing template path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Template not found"):
            _load_template("/nonexistent/template.png")

    def test_loads_valid_template(self, tmp_path: Path) -> None:
        """A valid PNG is loaded as a numpy array."""
        path = _red_square_template(tmp_path)
        template = _load_template(path)
        assert isinstance(template, np.ndarray)
        assert template.ndim == 3

    def test_invalid_image_raises_value_error(self, tmp_path: Path) -> None:
        """A non-image file raises ValueError."""
        fake = tmp_path / "not_an_image.png"
        fake.write_bytes(b"not image data")
        with pytest.raises(ValueError, match="Could not decode"):
            _load_template(fake)


# ---------------------------------------------------------------------------
# match_template stubs
# ---------------------------------------------------------------------------


class TestMatchTemplate:
    """Tests for the match_template function (stub behaviour)."""

    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        """match_template raises NotImplementedError (stub)."""
        path = _red_square_template(tmp_path)
        with pytest.raises(NotImplementedError):
            match_template(_white_image(), path)

    # -----------------------------------------------------------------------
    # Contract tests — verify expected behaviour once implemented
    # -----------------------------------------------------------------------

    @pytest.mark.skip(reason="Requires implementation")
    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        """Returns None when template is not present in image."""
        path = _red_square_template(tmp_path)
        result = match_template(_white_image(), path, threshold=0.95)
        assert result is None

    @pytest.mark.skip(reason="Requires implementation")
    def test_returns_find_result_on_match(self, tmp_path: Path) -> None:
        """Returns a FindResult with method='template' and confidence >= threshold."""
        # Create image that contains a red square
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        # Paste red square at (50, 50)
        for x in range(50, 70):
            for y in range(50, 70):
                img.putpixel((x, y), (255, 0, 0))
        template_path = _red_square_template(tmp_path)
        result = match_template(img, template_path, threshold=0.8)
        assert result is not None
        assert isinstance(result, FindResult)
        assert result.method == "template"
        assert result.confidence >= 0.8
        assert result.bbox.x >= 0

    @pytest.mark.skip(reason="Requires implementation")
    def test_region_restricts_search(self, tmp_path: Path) -> None:
        """Setting region limits the search area."""
        path = _red_square_template(tmp_path)
        # Template is not in the restricted region → should return None
        result = match_template(
            _white_image(300, 200),
            path,
            threshold=0.8,
            region=BBox(200, 0, 100, 200),
        )
        assert result is None


class TestMatchTemplateMultiscale:
    """Tests for multiscale template matching stub."""

    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        """match_template_multiscale raises NotImplementedError (stub)."""
        path = _red_square_template(tmp_path)
        with pytest.raises(NotImplementedError):
            match_template_multiscale(_white_image(), path)
