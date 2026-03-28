"""OpenCV template matching — single-scale and multi-scale.

All public functions return :class:`~screenwalker.vision.screen_state.FindResult`
objects so they are drop-in replacements for the OCR finder in the engine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from screenwalker.vision.screen_state import BBox, FindResult


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR numpy array.

    Args:
        image: Source PIL Image (any mode).

    Returns:
        OpenCV-compatible BGR uint8 array.
    """
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _load_template(template_path: Path | str) -> np.ndarray:
    """Load a template image from disk as a BGR numpy array.

    Args:
        template_path: Path to the template PNG/JPEG file.

    Returns:
        Template image as BGR uint8 array.

    Raises:
        FileNotFoundError: If the template file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    template = cv2.imread(str(path))
    if template is None:
        raise ValueError(f"Could not decode template image: {path}")
    return template


def match_template(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    region: BBox | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> FindResult | None:
    """Single-scale template matching using OpenCV.

    Args:
        image: Screenshot to search.
        template_path: Path to the reference template image.
        threshold: Minimum normalised match score (0.0–1.0).
        region: Optional bounding box to crop before matching.
        method: OpenCV matching method constant (default TM_CCOEFF_NORMED).

    Returns:
        :class:`FindResult` for the best match above threshold, or None.

    Example:
        >>> result = match_template(screenshot, "templates/ok_button.png")
        >>> if result:
        ...     print(result.bbox.center)
    """
    # TODO: implement using cv2.matchTemplate
    #   1. Convert image to grayscale cv array (optionally crop region)
    #   2. Load and convert template to grayscale
    #   3. cv2.matchTemplate → result matrix
    #   4. cv2.minMaxLoc → (min_val, max_val, min_loc, max_loc)
    #   5. If max_val >= threshold: build BBox, return FindResult
    raise NotImplementedError("TODO: implement match_template")


def match_template_all(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    region: BBox | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
    max_results: int = 50,
) -> list[FindResult]:
    """Find ALL non-overlapping occurrences of a template in an image.

    Uses non-maximum suppression to eliminate overlapping matches.

    Args:
        image: Screenshot to search.
        template_path: Path to the reference template image.
        threshold: Minimum match score.
        region: Optional search region.
        method: OpenCV matching method constant.
        max_results: Maximum number of results to return.

    Returns:
        List of :class:`FindResult` sorted by confidence descending.
    """
    # TODO: run matchTemplate, threshold result matrix, apply NMS
    raise NotImplementedError("TODO: implement match_template_all")


def match_template_multiscale(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    scale_min: float = 0.7,
    scale_max: float = 1.3,
    scale_steps: int = 10,
    region: BBox | None = None,
) -> FindResult | None:
    """Multi-scale template matching.

    Searches at multiple template sizes to handle zoom/DPI differences between
    the reference screenshot and the live screen.

    Args:
        image: Screenshot to search.
        template_path: Path to the reference template image.
        threshold: Minimum match score.
        scale_min: Smallest scale factor relative to original template.
        scale_max: Largest scale factor relative to original template.
        scale_steps: Number of evenly-spaced scale levels to try.
        region: Optional search region.

    Returns:
        :class:`FindResult` for the best match across all scales, or None.
    """
    # TODO: for each scale in np.linspace(scale_min, scale_max, scale_steps):
    #   1. Resize template
    #   2. Run single-scale match
    #   3. Track best result across scales
    raise NotImplementedError("TODO: implement match_template_multiscale")
