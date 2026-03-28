"""Screen-state identification and unified FindResult type.

:class:`FindResult` is the single data structure returned by every vision
method (OCR, template matching, YOLO).  :class:`ScreenStateDetector` combines
multiple finders to identify which "logical screen" the UI is currently on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in screen coordinates.

    Attributes:
        x: Left edge (pixels from left of screen/region).
        y: Top edge (pixels from top of screen/region).
        w: Width in pixels.
        h: Height in pixels.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        """Pixel coordinates of the bounding-box centre."""
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def right(self) -> int:
        """Right edge pixel coordinate."""
        return self.x + self.w

    @property
    def bottom(self) -> int:
        """Bottom edge pixel coordinate."""
        return self.y + self.h

    def offset(self, dx: int, dy: int) -> "BBox":
        """Return a new BBox shifted by (dx, dy).

        Args:
            dx: Horizontal pixel offset.
            dy: Vertical pixel offset.

        Returns:
            New translated BBox (original unchanged).
        """
        return BBox(self.x + dx, self.y + dy, self.w, self.h)


@dataclass(frozen=True)
class FindResult:
    """Unified result returned by every vision finder.

    Attributes:
        element: Human-readable description of the found element (text, label, etc.).
        confidence: Confidence score in [0.0, 1.0].
        bbox: Bounding box of the found element in screen coordinates.
        method: Vision method that produced this result (``ocr``, ``template``, ``yolo``).
        metadata: Optional extra data (e.g. raw OCR boxes, YOLO class id).
    """

    element: str
    confidence: float
    bbox: BBox
    method: str
    metadata: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Convenience accessor for bbox.center."""
        return self.bbox.center


class Finder(Protocol):
    """Protocol that all vision finders must implement.

    Any class implementing this protocol can be used interchangeably by the
    engine to locate UI elements.
    """

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.80,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Attempt to locate *query* within *image*.

        Args:
            image: Screenshot (full screen or cropped region).
            query: Search term — text string for OCR, image path for template.
            threshold: Minimum confidence score to accept a match.
            region: Optional bounding box restricting the search area.

        Returns:
            :class:`FindResult` if found above threshold, else None.
        """
        ...

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.80,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Find all matching occurrences of *query* in *image*.

        Args:
            image: Screenshot to search.
            query: Search term.
            threshold: Minimum confidence score.
            region: Optional bounding box restricting the search area.

        Returns:
            List of :class:`FindResult` sorted by confidence (highest first).
        """
        ...


class ScreenStateDetector:
    """Identifies the current "logical screen" by combining multiple finders.

    A screen state is defined by a set of *anchors* — OCR phrases or template
    images that must be present simultaneously.  The detector scores each
    registered state and returns the one with the highest combined confidence.

    Example:
        >>> detector = ScreenStateDetector()
        >>> detector.register("login_screen", ocr_anchors=["Sign In", "Forgot Password"])
        >>> detector.register("dashboard", template_anchors=["templates/logo.png"])
        >>> state = detector.identify(screenshot)
    """

    def __init__(self) -> None:
        """Initialize with an empty state registry."""
        # TODO: inject OCR finder and template finder
        self._states: dict[str, dict] = {}

    def register(
        self,
        state_id: str,
        ocr_anchors: list[str] | None = None,
        template_anchors: list[str] | None = None,
        threshold: float = 0.75,
    ) -> None:
        """Register a named screen state with its identifying anchors.

        Args:
            state_id: Unique identifier for the screen state.
            ocr_anchors: List of text strings that must be visible.
            template_anchors: List of template image paths that must be visible.
            threshold: Minimum average confidence to consider a state active.
        """
        self._states[state_id] = {
            "ocr_anchors": ocr_anchors or [],
            "template_anchors": template_anchors or [],
            "threshold": threshold,
        }

    def identify(self, image: Image.Image) -> str | None:
        """Identify the current screen state from a screenshot.

        Args:
            image: Full-screen or window screenshot.

        Returns:
            The ID of the best-matching registered state, or None if no state
            exceeds its threshold.
        """
        # TODO: score each registered state against image using finders
        raise NotImplementedError("TODO: implement screen state identification")

    def load_from_config(self, states_config: dict) -> None:
        """Bulk-register states from a config dictionary.

        Args:
            states_config: Mapping of state_id → anchor config dicts.
        """
        # TODO: iterate states_config and call self.register for each
        raise NotImplementedError("TODO: implement load_from_config")
