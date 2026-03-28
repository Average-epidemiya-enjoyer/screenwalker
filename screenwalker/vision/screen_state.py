"""Screen-state identification and unified FindResult type.

:class:`FindResult` is the single data structure returned by every vision
method (OCR, template matching, YOLO).  :class:`ScreenStateDetector` combines
multiple finders to identify which "logical screen" the UI is currently on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from PIL import Image
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from screenwalker.vision.ocr import OCRResult

log = structlog.get_logger(__name__)


# ── Shared geometry ──────────────────────────────────────────────────────────


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

    def offset(self, dx: int, dy: int) -> BBox:
        """Return a new BBox shifted by (dx, dy).

        Args:
            dx: Horizontal pixel offset.
            dy: Vertical pixel offset.

        Returns:
            New translated BBox (original unchanged).
        """
        return BBox(self.x + dx, self.y + dy, self.w, self.h)


# ── Unified find result ───────────────────────────────────────────────────────


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


# ── Finder protocol ───────────────────────────────────────────────────────────


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


# ── Screen fingerprint (Pydantic, YAML-serialisable) ─────────────────────────


class ScreenFingerprint(BaseModel):
    """Declarative descriptor for a named logical screen.

    Each fingerprint is matched against the current screenshot at runtime.
    Texts are compared with fuzzy matching; templates are matched with
    OpenCV template matching.

    Attributes:
        screen_id: Unique identifier for the screen (e.g. ``"notepad_main"``).
        required_texts: Texts that MUST be present.  Absence of any text
            penalises the score by ``-1.0``.
        forbidden_texts: Texts that must NOT be present.  Presence of any
            penalises the score by ``-1.0``.
        required_templates: Template image names that should be visible.
            Each match adds ``+0.5`` to the score.
        optional_texts: Texts whose presence adds ``+0.3`` (bonus only).
        match_threshold: Minimum normalised confidence to accept as a match.
    """

    screen_id: str
    required_texts: list[str] = Field(default_factory=list)
    forbidden_texts: list[str] = Field(default_factory=list)
    required_templates: list[str] = Field(default_factory=list)
    optional_texts: list[str] = Field(default_factory=list)
    match_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


# ── Identification result ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenIdentification:
    """Result of :meth:`ScreenStateAnalyzer.identify`.

    Attributes:
        screen_id: ID of the best-matching fingerprint, or ``None``.
        confidence: Normalised score in [0.0, 1.0].
        matched_texts: Required/optional texts that were found.
        matched_templates: Template names that were found.
        ocr_results: Raw OCR results used during identification.
        is_popup: True when a popup was detected on top of the identified screen.
    """

    screen_id: str | None
    confidence: float
    matched_texts: list[str] = field(default_factory=list)
    matched_templates: list[str] = field(default_factory=list)
    ocr_results: list[Any] = field(default_factory=list)
    is_popup: bool = False


# ── Popup info ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PopupInfo:
    """Result of :meth:`ScreenStateAnalyzer.detect_popup`.

    Attributes:
        popup_type: One of ``"error"``, ``"confirm"``, ``"loading"``.
        confidence: Normalised match score.
        matched_texts: Popup texts that were found.
    """

    popup_type: str
    confidence: float
    matched_texts: list[str] = field(default_factory=list)


# ── Built-in popup fingerprints ───────────────────────────────────────────────

_POPUP_FINGERPRINTS: list[ScreenFingerprint] = [
    ScreenFingerprint(
        screen_id="error_popup",
        required_texts=["Error"],
        optional_texts=["OK", "Close", "Cancel", "failed", "cannot", "unable"],
        match_threshold=0.5,
    ),
    ScreenFingerprint(
        screen_id="confirm_popup",
        required_texts=["OK", "Cancel"],
        optional_texts=["Yes", "No", "Confirm", "Are you sure"],
        match_threshold=0.6,
    ),
    ScreenFingerprint(
        screen_id="loading_popup",
        required_texts=["Please wait"],
        optional_texts=["Loading", "Processing", "…", "...", "progress"],
        match_threshold=0.4,
    ),
]


# ── Core analyser ─────────────────────────────────────────────────────────────


class ScreenStateAnalyzer:
    """Identifies the current logical screen by scoring registered fingerprints.

    The analyser runs OCR on the screenshot once and reuses the results across
    all fingerprint checks.  Template matching is performed only when a
    :class:`~screenwalker.vision.template_match.TemplateMatcher` is provided.

    Scoring per fingerprint:

    * Required text present  → ``+1.0``
    * Required text absent   → ``-1.0`` (raw score may go negative)
    * Forbidden text present → ``-1.0``
    * Required template found → ``+0.5``
    * Optional text present  → ``+0.3``

    The raw score is normalised against the maximum achievable score for that
    fingerprint.  The fingerprint with the highest normalised score above its
    ``match_threshold`` wins.

    Args:
        fingerprints: List of :class:`ScreenFingerprint` objects to match against.
        ocr_engine: An OCR engine instance (must implement ``recognize()``).
            Pass ``None`` to disable OCR-based matching.
        template_matcher: A :class:`~screenwalker.vision.template_match.TemplateMatcher`
            instance.  Pass ``None`` to disable template matching.
        fuzzy_threshold: Minimum ``rapidfuzz.partial_ratio`` score (0–100) used
            when comparing OCR texts against fingerprint texts.
    """

    def __init__(
        self,
        fingerprints: list[ScreenFingerprint] | None = None,
        ocr_engine: Any = None,
        template_matcher: Any = None,
        fuzzy_threshold: int = 70,
    ) -> None:
        self._fingerprints: list[ScreenFingerprint] = fingerprints or []
        self._ocr = ocr_engine
        self._matcher = template_matcher
        self._fuzzy_threshold = fuzzy_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def identify(self, screenshot: Image.Image) -> ScreenIdentification:
        """Identify the current screen from a screenshot.

        Args:
            screenshot: Full-screen or window screenshot.

        Returns:
            :class:`ScreenIdentification` for the best-matching fingerprint.
            If no fingerprint exceeds its threshold, ``screen_id`` is ``None``
            and ``confidence`` is ``0.0``.
        """
        ocr_results = self._run_ocr(screenshot)
        ocr_texts = self._collect_texts(ocr_results)

        best_id: str | None = None
        best_conf: float = 0.0
        best_texts: list[str] = []
        best_templates: list[str] = []

        for fp in self._fingerprints:
            conf, texts, templates = self._score(fp, screenshot, ocr_texts)
            log.debug(
                "screen_state.scored",
                screen_id=fp.screen_id,
                confidence=round(conf, 3),
            )
            if conf >= fp.match_threshold and conf > best_conf:
                best_id = fp.screen_id
                best_conf = conf
                best_texts = texts
                best_templates = templates

        popup = self.detect_popup(screenshot, ocr_results=ocr_results)

        return ScreenIdentification(
            screen_id=best_id,
            confidence=best_conf,
            matched_texts=best_texts,
            matched_templates=best_templates,
            ocr_results=ocr_results,
            is_popup=popup is not None,
        )

    def verify(self, screenshot: Image.Image, expected_screen_id: str) -> bool:
        """Check whether the screenshot matches a specific screen.

        Args:
            screenshot: Screenshot to evaluate.
            expected_screen_id: The ``screen_id`` to verify against.

        Returns:
            ``True`` if the identified screen matches *expected_screen_id*.
        """
        result = self.identify(screenshot)
        return result.screen_id == expected_screen_id

    def detect_popup(
        self,
        screenshot: Image.Image,
        *,
        ocr_results: list[Any] | None = None,
    ) -> PopupInfo | None:
        """Detect whether a popup dialog is currently shown.

        Checks the built-in popup fingerprints (error, confirm, loading).

        Args:
            screenshot: Screenshot to analyse.
            ocr_results: Pre-computed OCR results to reuse (avoids a second
                OCR pass when called from :meth:`identify`).

        Returns:
            :class:`PopupInfo` for the best-matching popup type, or ``None``
            if no popup is detected.
        """
        if ocr_results is None:
            ocr_results = self._run_ocr(screenshot)
        ocr_texts = self._collect_texts(ocr_results)

        best_popup: PopupInfo | None = None
        best_conf: float = 0.0

        for fp in _POPUP_FINGERPRINTS:
            conf, texts, _ = self._score(fp, screenshot, ocr_texts)
            if conf >= fp.match_threshold and conf > best_conf:
                best_conf = conf
                popup_type = fp.screen_id.replace("_popup", "")
                best_popup = PopupInfo(
                    popup_type=popup_type,
                    confidence=conf,
                    matched_texts=texts,
                )

        return best_popup

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_ocr(self, screenshot: Image.Image) -> list[Any]:
        """Run OCR and return word-level results (empty list if no engine)."""
        if self._ocr is None:
            return []
        try:
            return self._ocr.recognize(screenshot)
        except Exception:
            log.warning("screen_state.ocr_failed", exc_info=True)
            return []

    def _collect_texts(self, ocr_results: list[Any]) -> list[str]:
        """Build a flat list of searchable strings from OCR results.

        Includes individual word texts, line-grouped texts, and the full
        concatenated string — enabling both word-level and phrase-level
        fingerprint matching.
        """
        if not ocr_results:
            return []

        word_texts = [r.text for r in ocr_results if r.text.strip()]
        all_texts: list[str] = list(word_texts)

        # Add line-grouped texts when the OCR engine supports grouping
        if self._ocr is not None and hasattr(self._ocr, "_group_into_lines"):
            try:
                line_results = self._ocr._group_into_lines(ocr_results)
                for r in line_results:
                    if r.text.strip() and r.text not in all_texts:
                        all_texts.append(r.text)
            except Exception:
                log.warning("screen_state.line_grouping_failed", exc_info=True)

        full = " ".join(word_texts)
        if full and full not in all_texts:
            all_texts.append(full)

        return all_texts

    def _text_found(self, query: str, ocr_texts: list[str]) -> bool:
        """Return True if *query* is found (fuzzy) in any of *ocr_texts*."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            # Exact substring fallback
            q = query.lower()
            return any(q in t.lower() for t in ocr_texts)

        threshold = self._fuzzy_threshold
        for text in ocr_texts:
            if fuzz.partial_ratio(query.lower(), text.lower()) >= threshold:
                return True
        return False

    def _score(
        self,
        fp: ScreenFingerprint,
        screenshot: Image.Image,
        ocr_texts: list[str],
    ) -> tuple[float, list[str], list[str]]:
        """Score *fp* against the current screenshot and OCR texts.

        Returns:
            Tuple of (normalised_confidence, matched_texts, matched_templates).
        """
        raw: float = 0.0
        matched_texts: list[str] = []
        matched_templates: list[str] = []

        for text in fp.required_texts:
            if self._text_found(text, ocr_texts):
                raw += 1.0
                matched_texts.append(text)
            else:
                raw -= 1.0

        for text in fp.forbidden_texts:
            if self._text_found(text, ocr_texts):
                raw -= 1.0

        if self._matcher is not None:
            for tmpl in fp.required_templates:
                try:
                    result = self._matcher.find_one(screenshot, tmpl, threshold=0.75)
                    if result is not None:
                        raw += 0.5
                        matched_templates.append(tmpl)
                except Exception:
                    log.warning(
                        "screen_state.template_match_failed",
                        template=tmpl,
                        exc_info=True,
                    )

        for text in fp.optional_texts:
            if self._text_found(text, ocr_texts):
                raw += 0.3
                matched_texts.append(text)

        max_score = (
            len(fp.required_texts) * 1.0
            + len(fp.required_templates) * 0.5
            + len(fp.optional_texts) * 0.3
        )

        if max_score > 0:
            confidence = max(0.0, min(1.0, raw / max_score))
        else:
            confidence = 1.0 if raw >= 0 else 0.0

        return confidence, matched_texts, matched_templates


# ── Backward-compatible detector ──────────────────────────────────────────────


class ScreenStateDetector:
    """Identifies the current "logical screen" by combining multiple finders.

    Wraps :class:`ScreenStateAnalyzer` and exposes the legacy ``register`` /
    ``identify`` / ``load_from_config`` API so existing call sites keep working.

    Args:
        ocr_engine: Optional OCR engine for text-based anchors.
        template_matcher: Optional template matcher for image-based anchors.

    Example:
        >>> detector = ScreenStateDetector()
        >>> detector.register("login_screen", ocr_anchors=["Sign In"])
        >>> state = detector.identify(screenshot)
    """

    def __init__(
        self,
        ocr_engine: Any = None,
        template_matcher: Any = None,
        fuzzy_threshold: int = 70,
    ) -> None:
        self._ocr = ocr_engine
        self._matcher = template_matcher
        self._fuzzy_threshold = fuzzy_threshold
        self._fingerprints: list[ScreenFingerprint] = []
        self._analyzer: ScreenStateAnalyzer | None = None

    def _get_analyzer(self) -> ScreenStateAnalyzer:
        """Lazily create the inner :class:`ScreenStateAnalyzer`.

        The cached instance is invalidated (set to ``None``) by :meth:`register`
        and :meth:`load_from_config` whenever the fingerprint list changes.
        """
        if self._analyzer is None:
            self._analyzer = ScreenStateAnalyzer(
                fingerprints=self._fingerprints,
                ocr_engine=self._ocr,
                template_matcher=self._matcher,
                fuzzy_threshold=self._fuzzy_threshold,
            )
        return self._analyzer

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
        fp = ScreenFingerprint(
            screen_id=state_id,
            required_texts=ocr_anchors or [],
            required_templates=template_anchors or [],
            match_threshold=threshold,
        )
        self._fingerprints.append(fp)
        self._analyzer = None  # invalidate cached analyzer

    def identify(self, image: Image.Image) -> str | None:
        """Identify the current screen state from a screenshot.

        Args:
            image: Full-screen or window screenshot.

        Returns:
            The ID of the best-matching registered state, or None if no state
            exceeds its threshold.
        """
        result = self._get_analyzer().identify(image)
        return result.screen_id

    def load_from_config(self, states_config: dict) -> None:
        """Bulk-register states from a config dictionary.

        Expected format::

            {
                "login_screen": {
                    "required_texts": ["Sign In"],
                    "forbidden_texts": [],
                    "required_templates": [],
                    "optional_texts": ["Forgot Password"],
                    "match_threshold": 0.75
                },
                ...
            }

        Args:
            states_config: Mapping of ``state_id`` → fingerprint field dicts.
        """
        for state_id, cfg in states_config.items():
            try:
                fp = ScreenFingerprint(
                    screen_id=state_id,
                    required_texts=cfg.get("required_texts", []),
                    forbidden_texts=cfg.get("forbidden_texts", []),
                    required_templates=cfg.get("required_templates", []),
                    optional_texts=cfg.get("optional_texts", []),
                    match_threshold=cfg.get("match_threshold", 0.75),
                )
            except Exception:
                log.warning(
                    "screen_state.invalid_fingerprint_skipped",
                    state_id=state_id,
                    exc_info=True,
                )
                continue
            self._fingerprints.append(fp)
        self._analyzer = None  # invalidate cached analyzer
