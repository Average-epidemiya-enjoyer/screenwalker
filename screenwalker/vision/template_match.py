"""OpenCV template matching — single-scale, multi-scale, and NMS.

Public API:
- :class:`TemplateMatcher` — class-based API with template directory + caching.
- :func:`match_template` / :func:`match_template_all` /
  :func:`match_template_multiscale` — standalone functions for one-off use.
- :func:`_nms` — pure-numpy Non-Maximum Suppression (exported for testing).
- :func:`draw_matches` — debug visualisation.
- :func:`capture_template` — interactive region capture helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import structlog
from PIL import Image, ImageDraw

from screenwalker.vision.screen_state import BBox, FindResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversion helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR uint8 array.

    Args:
        image: Source PIL Image (any mode).

    Returns:
        OpenCV-compatible BGR uint8 ndarray of shape (H, W, 3).
    """
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR array to grayscale.

    Args:
        bgr: BGR uint8 array of shape (H, W, 3).

    Returns:
        Grayscale uint8 array of shape (H, W).
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _load_template(template_path: Path | str) -> np.ndarray:
    """Load a template image from disk as a BGR uint8 array.

    Args:
        template_path: Path to the template PNG/JPEG file.

    Returns:
        Template image as BGR uint8 ndarray.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    template = cv2.imread(str(path))
    if template is None:
        raise ValueError(f"Could not decode template image: {path}")
    return template


# ---------------------------------------------------------------------------
# MatchResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """Result of a single template match.

    Attributes:
        template_name: Name (stem) of the matched template.
        confidence: Normalised match score in [0.0, 1.0].
        center: Pixel coordinates of the match centre ``(x, y)``.
        bbox: Bounding box of the match in image coordinates.
        scale: Template scale that produced this match (1.0 = original size).
    """

    template_name: str
    confidence: float
    center: tuple[int, int]
    bbox: BBox
    scale: float = 1.0


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]:
    """Pure-numpy Non-Maximum Suppression.

    Args:
        boxes: Array of shape ``(N, 4)`` with columns ``[x, y, w, h]``.
        scores: Array of shape ``(N,)`` with confidence scores.
        iou_threshold: Boxes whose IoU with the currently selected box
            exceeds this value are suppressed.

    Returns:
        List of kept indices into *boxes*, sorted by score descending.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = (boxes[:, 0] + boxes[:, 2]).astype(float)
    y2 = (boxes[:, 1] + boxes[:, 3]).astype(float)
    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-9)

        order = rest[iou <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# TemplateMatcher class
# ---------------------------------------------------------------------------


class TemplateMatcher:
    """Class-based template matcher with template-directory management and caching.

    Loaded templates are cached in memory so repeated calls to
    :meth:`find_one` / :meth:`find_all` for the same template name do not
    re-read from disk.

    Attributes:
        templates_dir: Root directory that contains reference PNG templates.
        method: OpenCV matching method (default ``cv2.TM_CCOEFF_NORMED``).
    """

    def __init__(
        self,
        templates_dir: Path,
        method: int = cv2.TM_CCOEFF_NORMED,
    ) -> None:
        """Initialise TemplateMatcher.

        Args:
            templates_dir: Directory containing ``.png`` template files.
            method: OpenCV ``matchTemplate`` method constant.
        """
        self.templates_dir = Path(templates_dir)
        self.method = method
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def load_template(self, name: str) -> np.ndarray:
        """Load a template by name, with in-memory caching.

        Tries ``templates_dir/<name>`` first, then ``templates_dir/<name>.png``.

        Args:
            name: Template stem (e.g. ``"ok_button"``) or full filename.

        Returns:
            BGR uint8 ndarray.

        Raises:
            FileNotFoundError: If neither candidate path exists.
        """
        if name in self._cache:
            return self._cache[name]

        for candidate in [name, f"{name}.png"]:
            path = self.templates_dir / candidate
            if path.exists():
                self._cache[name] = _load_template(path)
                logger.debug("Loaded template", name=name, path=str(path))
                return self._cache[name]

        raise FileNotFoundError(
            f"Template '{name}' not found in {self.templates_dir}. "
            f"Tried: {name}, {name}.png"
        )

    def clear_cache(self) -> None:
        """Remove all cached templates from memory."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Single-scale matching
    # ------------------------------------------------------------------

    def find_one(
        self,
        screenshot: Image.Image,
        template_name: str,
        threshold: float = 0.8,
        region: BBox | None = None,
    ) -> MatchResult | None:
        """Find the single best match of a template in a screenshot.

        Args:
            screenshot: Full-screen or window PIL Image.
            template_name: Template stem to load from :attr:`templates_dir`.
            threshold: Minimum normalised match score.
            region: Optional bounding box to restrict the search area.

        Returns:
            :class:`MatchResult` for the best match, or None.
        """
        tmpl = self.load_template(template_name)
        gray_screen, ox, oy = self._prepare_screen(screenshot, region)
        gray_tmpl = _to_gray(tmpl)

        hit = self._match_best(gray_screen, gray_tmpl, threshold)
        if hit is None:
            return None

        conf, x, y = hit
        th, tw = gray_tmpl.shape[:2]
        logger.debug("Template found", name=template_name, confidence=conf, x=ox + x, y=oy + y)
        return MatchResult(
            template_name=template_name,
            confidence=conf,
            center=(ox + x + tw // 2, oy + y + th // 2),
            bbox=BBox(x=ox + x, y=oy + y, w=tw, h=th),
            scale=1.0,
        )

    def find_all(
        self,
        screenshot: Image.Image,
        template_name: str,
        threshold: float = 0.8,
        region: BBox | None = None,
    ) -> list[MatchResult]:
        """Find all non-overlapping matches of a template in a screenshot.

        Applies Non-Maximum Suppression (IoU > 0.5) to remove duplicate
        detections from overlapping match-map peaks.

        Args:
            screenshot: Full-screen or window PIL Image.
            template_name: Template stem to load.
            threshold: Minimum match score.
            region: Optional search region.

        Returns:
            List of :class:`MatchResult` sorted by confidence descending.
        """
        tmpl = self.load_template(template_name)
        gray_screen, ox, oy = self._prepare_screen(screenshot, region)
        gray_tmpl = _to_gray(tmpl)

        th, tw = gray_tmpl.shape[:2]
        raw = self._match_all(gray_screen, gray_tmpl, threshold)
        if not raw:
            return []

        boxes = np.array([[x, y, tw, th] for _, x, y in raw], dtype=float)
        scores = np.array([c for c, _, _ in raw])
        keep = _nms(boxes, scores, iou_threshold=0.5)

        matches = [
            MatchResult(
                template_name=template_name,
                confidence=raw[i][0],
                center=(ox + raw[i][1] + tw // 2, oy + raw[i][2] + th // 2),
                bbox=BBox(x=ox + raw[i][1], y=oy + raw[i][2], w=tw, h=th),
                scale=1.0,
            )
            for i in keep
        ]
        logger.debug(
            "Template find_all", name=template_name, count=len(matches)
        )
        return sorted(matches, key=lambda r: r.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Multi-scale matching
    # ------------------------------------------------------------------

    def find_one_multiscale(
        self,
        screenshot: Image.Image,
        template_name: str,
        scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
        threshold: float = 0.75,
        region: BBox | None = None,
    ) -> MatchResult | None:
        """Find the best match across multiple template scales.

        Resizes the *template* (not the screenshot) at each scale level and
        picks the scale that gives the highest confidence above *threshold*.
        Useful for handling DPI / resolution differences between the reference
        screenshot and the live screen.

        Args:
            screenshot: Full-screen or window PIL Image.
            template_name: Template stem to load.
            scales: Tuple of scale factors to try (relative to original size).
            threshold: Minimum match score to consider a hit.
            region: Optional search region.

        Returns:
            :class:`MatchResult` for the best match across all scales, or None.
        """
        tmpl = self.load_template(template_name)
        gray_screen, ox, oy = self._prepare_screen(screenshot, region)
        gray_tmpl_orig = _to_gray(tmpl)
        h_orig, w_orig = gray_tmpl_orig.shape[:2]

        best: MatchResult | None = None

        for scale in scales:
            new_w = max(1, int(w_orig * scale))
            new_h = max(1, int(h_orig * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            gray_tmpl = cv2.resize(
                gray_tmpl_orig, (new_w, new_h), interpolation=interp
            )

            hit = self._match_best(gray_screen, gray_tmpl, threshold)
            if hit is None:
                continue

            conf, x, y = hit
            candidate = MatchResult(
                template_name=template_name,
                confidence=conf,
                center=(ox + x + new_w // 2, oy + y + new_h // 2),
                bbox=BBox(x=ox + x, y=oy + y, w=new_w, h=new_h),
                scale=scale,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        if best is not None:
            logger.debug(
                "Multiscale match",
                name=template_name,
                scale=best.scale,
                confidence=best.confidence,
            )
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_screen(
        self,
        screenshot: Image.Image,
        region: BBox | None,
    ) -> tuple[np.ndarray, int, int]:
        """Return ``(gray_array, offset_x, offset_y)``."""
        if region is not None:
            src = screenshot.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            ox, oy = region.x, region.y
        else:
            src = screenshot
            ox, oy = 0, 0
        return _to_gray(_pil_to_cv(src)), ox, oy

    def _match_best(
        self,
        gray_screen: np.ndarray,
        gray_tmpl: np.ndarray,
        threshold: float,
    ) -> tuple[float, int, int] | None:
        """Return ``(conf, x, y)`` for the best match, or None."""
        if (
            gray_tmpl.shape[0] > gray_screen.shape[0]
            or gray_tmpl.shape[1] > gray_screen.shape[1]
        ):
            return None
        result = cv2.matchTemplate(gray_screen, gray_tmpl, self.method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return float(max_val), int(max_loc[0]), int(max_loc[1])
        return None

    def _match_all(
        self,
        gray_screen: np.ndarray,
        gray_tmpl: np.ndarray,
        threshold: float,
    ) -> list[tuple[float, int, int]]:
        """Return all ``(conf, x, y)`` hits above *threshold*."""
        if (
            gray_tmpl.shape[0] > gray_screen.shape[0]
            or gray_tmpl.shape[1] > gray_screen.shape[1]
        ):
            return []
        result = cv2.matchTemplate(gray_screen, gray_tmpl, self.method)
        ys, xs = np.where(result >= threshold)
        return [(float(result[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def draw_matches(
    screenshot: Image.Image,
    matches: list[MatchResult],
    box_color: tuple[int, int, int] = (0, 220, 0),
    label_color: tuple[int, int, int] = (255, 50, 50),
    line_width: int = 2,
) -> Image.Image:
    """Draw bounding boxes and labels on a copy of *screenshot*.

    Args:
        screenshot: Source image (not modified in place).
        matches: List of :class:`MatchResult` to draw.
        box_color: RGB colour for the bounding-box outline.
        label_color: RGB colour for the confidence label.
        line_width: Outline thickness in pixels.

    Returns:
        New PIL Image with annotations.
    """
    img = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for m in matches:
        b = m.bbox
        draw.rectangle(
            [b.x, b.y, b.x + b.w, b.y + b.h],
            outline=box_color,
            width=line_width,
        )
        label = f"{m.template_name} {m.confidence:.2f}"
        draw.text((b.x + 2, max(0, b.y - 12)), label, fill=label_color)
    return img


def capture_template(
    name: str,
    region: BBox,
    templates_dir: Path | str = Path("templates"),
) -> Path:
    """Capture a screen region and save it as a template PNG.

    Takes a screenshot, crops it to *region*, and saves the result as
    ``<templates_dir>/<name>.png``.

    Args:
        name: Template stem (no extension).
        region: Screen region to capture, in absolute screen coordinates.
        templates_dir: Destination directory (created if it does not exist).

    Returns:
        Absolute path of the saved template file.
    """
    import pyautogui  # lazy to keep the module importable without a display

    dest = Path(templates_dir)
    dest.mkdir(parents=True, exist_ok=True)

    screenshot = pyautogui.screenshot()
    cropped = screenshot.crop(
        (region.x, region.y, region.right, region.bottom)
    )
    path = (dest / f"{name}.png").resolve()
    cropped.save(path, "PNG")
    logger.info("Saved template", name=name, path=str(path))
    return path


# ---------------------------------------------------------------------------
# Standalone functions (backward-compatible, path-based API)
# ---------------------------------------------------------------------------


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
        method: OpenCV matching method constant.

    Returns:
        :class:`FindResult` for the best match above threshold, or None.
    """
    path = Path(template_path)
    tmpl = _load_template(path)

    if region is not None:
        src = image.crop((region.x, region.y, region.right, region.bottom))
        ox, oy = region.x, region.y
    else:
        src = image
        ox, oy = 0, 0

    gray_screen = _to_gray(_pil_to_cv(src))
    gray_tmpl = _to_gray(tmpl)
    th, tw = gray_tmpl.shape[:2]

    if th > gray_screen.shape[0] or tw > gray_screen.shape[1]:
        return None

    result = cv2.matchTemplate(gray_screen, gray_tmpl, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    x, y = max_loc
    return FindResult(
        element=path.stem,
        confidence=float(max_val),
        bbox=BBox(x=ox + x, y=oy + y, w=tw, h=th),
        method="template",
        metadata={"scale": 1.0},
    )


def match_template_all(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    region: BBox | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
    max_results: int = 50,
) -> list[FindResult]:
    """Find ALL non-overlapping occurrences of a template in an image.

    Applies Non-Maximum Suppression (IoU > 0.5) to eliminate duplicate
    detections.

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
    path = Path(template_path)
    tmpl = _load_template(path)

    if region is not None:
        src = image.crop((region.x, region.y, region.right, region.bottom))
        ox, oy = region.x, region.y
    else:
        src = image
        ox, oy = 0, 0

    gray_screen = _to_gray(_pil_to_cv(src))
    gray_tmpl = _to_gray(tmpl)
    th, tw = gray_tmpl.shape[:2]

    if th > gray_screen.shape[0] or tw > gray_screen.shape[1]:
        return []

    result_map = cv2.matchTemplate(gray_screen, gray_tmpl, method)
    ys, xs = np.where(result_map >= threshold)
    raw = [(float(result_map[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
    if not raw:
        return []

    boxes = np.array([[x, y, tw, th] for _, x, y in raw], dtype=float)
    scores = np.array([c for c, _, _ in raw])
    keep = _nms(boxes, scores, iou_threshold=0.5)

    out: list[FindResult] = []
    for i in keep[:max_results]:
        conf, x, y = raw[i]
        out.append(
            FindResult(
                element=path.stem,
                confidence=conf,
                bbox=BBox(x=ox + x, y=oy + y, w=tw, h=th),
                method="template",
                metadata={"scale": 1.0},
            )
        )
    return sorted(out, key=lambda r: r.confidence, reverse=True)


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
    path = Path(template_path)
    tmpl = _load_template(path)

    if region is not None:
        src = image.crop((region.x, region.y, region.right, region.bottom))
        ox, oy = region.x, region.y
    else:
        src = image
        ox, oy = 0, 0

    gray_screen = _to_gray(_pil_to_cv(src))
    gray_tmpl_orig = _to_gray(tmpl)
    h_orig, w_orig = gray_tmpl_orig.shape[:2]

    best_conf = -1.0
    best: FindResult | None = None

    for scale in np.linspace(scale_min, scale_max, scale_steps):
        scale = float(scale)
        new_w = max(1, int(w_orig * scale))
        new_h = max(1, int(h_orig * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray_tmpl = cv2.resize(
            gray_tmpl_orig, (new_w, new_h), interpolation=interp
        )

        if new_h > gray_screen.shape[0] or new_w > gray_screen.shape[1]:
            continue

        result_map = cv2.matchTemplate(
            gray_screen, gray_tmpl, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result_map)

        if max_val >= threshold and max_val > best_conf:
            best_conf = max_val
            x, y = max_loc
            best = FindResult(
                element=path.stem,
                confidence=float(max_val),
                bbox=BBox(x=ox + x, y=oy + y, w=new_w, h=new_h),
                method="template",
                metadata={"scale": scale},
            )

    return best
