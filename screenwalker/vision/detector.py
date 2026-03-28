"""YOLO-based UI element detector.

Three variants are available, selected automatically by :func:`create_detector`:

* **UIElementDetector** — wraps any YOLO-compatible model (OmniParser V2,
  fine-tuned YOLOv8n, or any ``ultralytics``-compatible weights).  Requires
  the ``yolo`` extra: ``pip install screenwalker[yolo]``.

* **NullDetector** — returns empty results with no dependencies.  Used when
  no model is configured or ``ultralytics`` is not installed.

Recommended model: **OmniParser V2** from Hugging Face
(``microsoft/OmniParser-v2.0``, ``icon_detect/best.pt``).
Download with ``python scripts/download_model.py``.

Custom fine-tuning: ``python scripts/train_detector.py``.

Class normalisation
-------------------
Raw YOLO class names are normalised to a canonical set of UI element types:

    ``button``, ``text_field``, ``icon``, ``checkbox``, ``radio_button``,
    ``dropdown``, ``link``, ``image``, ``label``, ``toggle``, ``slider``

Unknown class names are kept as-is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from screenwalker.vision.screen_state import BBox, FindResult

if TYPE_CHECKING:
    from screenwalker.utils.config import VisionConfig


# ---------------------------------------------------------------------------
# Class taxonomy — raw model names → canonical UI element type
# ---------------------------------------------------------------------------

UI_CLASS_ALIASES: dict[str, list[str]] = {
    "button": ["button", "btn", "push_button", "icon_button", "clickable"],
    "text_field": ["text_field", "input", "textbox", "edit", "text_input", "entry", "edittext"],
    "icon": ["icon", "symbol", "glyph", "pictogram"],
    "checkbox": ["checkbox", "check_box", "check", "tick_box"],
    "radio_button": ["radio", "radio_button", "radiobutton", "option_button"],
    "dropdown": ["dropdown", "select", "combobox", "combo", "list_box", "spinner"],
    "link": ["link", "hyperlink", "url", "anchor", "a"],
    "image": ["image", "img", "picture", "photo", "bitmap"],
    "label": ["label", "text", "static_text", "static"],
    "toggle": ["toggle", "switch", "on_off"],
    "slider": ["slider", "range", "seekbar", "progress"],
}

# Reverse lookup: raw/alias → canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in UI_CLASS_ALIASES.items()
    for alias in aliases
}
# The canonical names themselves also map to themselves
for _c in list(UI_CLASS_ALIASES):
    _ALIAS_TO_CANONICAL[_c] = _c


def normalise_class(raw: str) -> str:
    """Map a raw model class name to its canonical UI element type.

    Args:
        raw: Class name as returned by the model (e.g. ``"btn"``, ``"input"``).

    Returns:
        Canonical type string, or *raw* if no mapping is known.
    """
    return _ALIAS_TO_CANONICAL.get(raw.lower().strip(), raw.lower().strip())


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionResult:
    """A single UI element detection.

    Attributes:
        class_name: Canonical UI element type (``"button"``, ``"text_field"``, …).
        confidence: Model confidence in ``[0.0, 1.0]``.
        bbox: Bounding box in screen coordinates.
    """

    class_name: str
    confidence: float
    bbox: BBox

    @property
    def center(self) -> tuple[int, int]:
        """Pixel coordinates of the detection centre."""
        return self.bbox.center

    def to_find_result(self) -> FindResult:
        """Convert to a :class:`~screenwalker.vision.screen_state.FindResult`.

        Returns:
            FindResult with method ``"yolo"`` and element equal to class_name.
        """
        return FindResult(
            element=self.class_name,
            confidence=self.confidence,
            bbox=self.bbox,
            method="yolo",
            metadata={"yolo_class": self.class_name},
        )


# ---------------------------------------------------------------------------
# UIElementDetector
# ---------------------------------------------------------------------------


class UIElementDetector:
    """Detect UI elements using a YOLO-compatible model.

    Supports any ``ultralytics``-compatible weights file, including
    OmniParser V2 and custom fine-tuned YOLOv8 models.

    Attributes:
        model_path: Path to the ``*.pt`` weights file, or ``None`` for
            no-op mode.
        confidence_threshold: Minimum detection confidence (0.0–1.0).
        class_names: Mapping ``{class_index → raw_class_name}`` read from the
            loaded model.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialise the detector.

        Does **not** fail if ``ultralytics`` is not installed — the
        ``available`` property will return ``False`` and all find calls
        will return empty results.

        Args:
            model_path: Path to YOLO weights (``*.pt``).  ``None`` keeps the
                detector in no-op mode.
            confidence_threshold: Minimum confidence to accept a detection.
        """
        self.model_path: Path | None = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self._model: Any = None
        self.class_names: dict[int, str] = {}

        if self.model_path is not None:
            self._load_model()

    @property
    def available(self) -> bool:
        """``True`` if a model is loaded and inference can run."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the YOLO model from :attr:`model_path`.

        Raises:
            ImportError: If ``ultralytics`` is not installed.
            FileNotFoundError: If the weights file does not exist.
        """
        if self.model_path is None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.model_path}\n"
                "Run: python scripts/download_model.py"
            )
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO detection: "
                "pip install screenwalker[yolo]"
            ) from exc

        self._model = YOLO(str(self.model_path))
        self.class_names = dict(self._model.names)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect(
        self,
        screenshot: Image.Image,
        region: BBox | None = None,
    ) -> list[DetectionResult]:
        """Detect all UI elements in *screenshot*.

        Args:
            screenshot: Full-screen or cropped PIL image.
            region: Optional bounding box restricting the inference area.
                Returned bounding boxes are translated back to full-image
                coordinates.

        Returns:
            List of :class:`DetectionResult` sorted by confidence descending.
            Empty if the model is not available.
        """
        if not self.available:
            return []

        search_image = screenshot
        offset_x, offset_y = 0, 0
        if region is not None:
            search_image = screenshot.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            offset_x, offset_y = region.x, region.y

        img_array = np.array(search_image.convert("RGB"))

        try:
            results = self._model.predict(
                img_array,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception:
            return []

        detections: list[DetectionResult] = []
        for result in results:
            for box in result.boxes:
                try:
                    cls_idx = int(box.cls.item())
                    raw_name = self.class_names.get(cls_idx, str(cls_idx))
                    class_name = normalise_class(raw_name)
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    bbox = BBox(
                        x=int(x1) + offset_x,
                        y=int(y1) + offset_y,
                        w=max(1, int(x2 - x1)),
                        h=max(1, int(y2 - y1)),
                    )
                    detections.append(DetectionResult(
                        class_name=class_name,
                        confidence=conf,
                        bbox=bbox,
                    ))
                except Exception:
                    continue

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Filtered search
    # ------------------------------------------------------------------

    def find_by_class(
        self,
        screenshot: Image.Image,
        element_class: str,
        region: BBox | None = None,
    ) -> list[DetectionResult]:
        """Detect all elements of a specific UI class.

        Both the raw model class name and the canonical alias are checked,
        so ``"btn"`` and ``"button"`` both match a button.

        Args:
            screenshot: Screenshot to analyse.
            element_class: UI element type to filter for (e.g. ``"button"``).
            region: Optional search region.

        Returns:
            Matching detections sorted by confidence descending.
        """
        canonical_target = normalise_class(element_class)
        all_detections = self.detect(screenshot, region=region)
        return [
            d for d in all_detections
            if d.class_name == canonical_target
            or normalise_class(d.class_name) == canonical_target
        ]

    def find_nearest(
        self,
        screenshot: Image.Image,
        element_class: str,
        anchor_text: str,
        ocr_engine: Any,
    ) -> DetectionResult | None:
        """Find the UI element of *element_class* nearest to *anchor_text*.

        Use case: locate the ``"button"`` closest to the text
        ``"Confirm deletion?"`` to click the right OK button in a dialog.

        Steps:

        1. Run OCR to locate *anchor_text* and get its centre coordinates.
        2. Detect all elements of *element_class* via YOLO.
        3. Return the detection with the smallest Euclidean distance from the
           anchor text centre.

        Args:
            screenshot: Screenshot to search.
            element_class: UI element type (e.g. ``"button"``).
            anchor_text: Text to use as a spatial reference point.
            ocr_engine: An OCR engine implementing ``find_text(image, query)``.

        Returns:
            :class:`DetectionResult` for the nearest matching element, or
            ``None`` if either the anchor text or any matching element is
            not found.
        """
        # Step 1 — find anchor text position via OCR
        anchor_result = ocr_engine.find_text(screenshot, anchor_text)
        if anchor_result is None:
            return None
        anchor_cx, anchor_cy = anchor_result.bbox.center

        # Step 2 — detect elements of the requested class
        candidates = self.find_by_class(screenshot, element_class)
        if not candidates:
            return None

        # Step 3 — return closest by Euclidean distance
        def _dist(det: DetectionResult) -> float:
            ex, ey = det.center
            return math.sqrt((ex - anchor_cx) ** 2 + (ey - anchor_cy) ** 2)

        return min(candidates, key=_dist)

    # ------------------------------------------------------------------
    # Finder Protocol implementation
    # ------------------------------------------------------------------

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Return the best-matching detection as a :class:`FindResult`.

        Implements the :class:`~screenwalker.vision.screen_state.Finder`
        protocol so the engine can use this detector interchangeably with
        OCR and template finders.

        Args:
            image: Screenshot to search.
            query: UI class label to search for (e.g. ``"button"``).
            threshold: Override :attr:`confidence_threshold` for this call.
            region: Optional search region.

        Returns:
            :class:`FindResult` for the best detection, or ``None``.
        """
        cutoff = threshold if threshold is not None else self.confidence_threshold
        results = self.find_by_class(image, query, region=region)
        if not results:
            return None
        best = results[0]  # already sorted by confidence
        if best.confidence < cutoff:
            return None
        return best.to_find_result()

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Return all matching detections as :class:`FindResult` objects.

        Args:
            image: Screenshot to search.
            query: UI class label.
            threshold: Override :attr:`confidence_threshold` for this call.
            region: Optional search region.

        Returns:
            List of :class:`FindResult` sorted by confidence descending.
        """
        cutoff = threshold if threshold is not None else self.confidence_threshold
        results = self.find_by_class(image, query, region=region)
        return [
            d.to_find_result()
            for d in results
            if d.confidence >= cutoff
        ]

    def detect_all(self, image: Image.Image) -> list[DetectionResult]:
        """Detect every UI element visible in *image* regardless of class.

        Useful for exploratory automation and screen-state identification.

        Args:
            image: Screenshot to analyse.

        Returns:
            All detected elements sorted by confidence descending.
        """
        return self.detect(image)


# ---------------------------------------------------------------------------
# NullDetector — no-op fallback
# ---------------------------------------------------------------------------


class NullDetector:
    """No-op detector that always returns empty results.

    Used when no model is configured or ``ultralytics`` is not installed.
    Satisfies the same interface as :class:`UIElementDetector`.

    Attributes:
        available: Always ``False``.
        class_names: Always empty.
    """

    available: bool = False
    class_names: dict[int, str] = {}

    def detect(self, screenshot: Image.Image, region: BBox | None = None) -> list[DetectionResult]:
        """Return an empty list (no model loaded)."""
        return []

    def find_by_class(
        self,
        screenshot: Image.Image,
        element_class: str,
        region: BBox | None = None,
    ) -> list[DetectionResult]:
        """Return an empty list (no model loaded)."""
        return []

    def find_nearest(
        self,
        screenshot: Image.Image,
        element_class: str,
        anchor_text: str,
        ocr_engine: Any,
    ) -> DetectionResult | None:
        """Return ``None`` (no model loaded)."""
        return None

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Return ``None`` (no model loaded)."""
        return None

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Return an empty list (no model loaded)."""
        return []

    def detect_all(self, image: Image.Image) -> list[DetectionResult]:
        """Return an empty list (no model loaded)."""
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_detector(config: Any) -> UIElementDetector | NullDetector:
    """Create a detector from :class:`~screenwalker.utils.config.VisionConfig`.

    Selection logic:

    1. If ``config.yolo_enabled`` is ``False`` → return :class:`NullDetector`.
    2. If ``config.yolo_model_path`` is set and the file exists →
       return :class:`UIElementDetector` with that path.
    3. If the default OmniParser path
       (``models/icon_detect/best.pt``) exists → use it.
    4. Otherwise → return :class:`NullDetector` and log a warning.

    Args:
        config: A :class:`~screenwalker.utils.config.VisionConfig` (or any
            object with ``yolo_enabled``, ``yolo_model_path``,
            ``yolo_confidence`` attributes).

    Returns:
        A ready-to-use detector (or :class:`NullDetector` if unavailable).
    """
    import structlog as _sl
    _log = _sl.get_logger(__name__)

    if not getattr(config, "yolo_enabled", False):
        return NullDetector()

    confidence = getattr(config, "yolo_confidence", 0.50)

    # Explicit model path from config
    model_path_raw: str | None = getattr(config, "yolo_model_path", None)
    if model_path_raw:
        model_path = Path(model_path_raw)
        if model_path.exists():
            try:
                return UIElementDetector(
                    model_path=model_path,
                    confidence_threshold=confidence,
                )
            except (ImportError, FileNotFoundError) as exc:
                _log.warning("YOLO detector init failed", error=str(exc))
                return NullDetector()
        else:
            _log.warning("YOLO model not found", path=str(model_path))

    # Default OmniParser location
    default_path = Path("models/icon_detect/best.pt")
    if default_path.exists():
        try:
            return UIElementDetector(
                model_path=default_path,
                confidence_threshold=confidence,
            )
        except (ImportError, FileNotFoundError) as exc:
            _log.warning("YOLO detector init failed (default path)", error=str(exc))

    _log.info(
        "YOLO enabled but no model found — using NullDetector. "
        "Run: python scripts/download_model.py"
    )
    return NullDetector()


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

#: Alias kept for existing call sites that reference YOLODetector by name.
YOLODetector = UIElementDetector
