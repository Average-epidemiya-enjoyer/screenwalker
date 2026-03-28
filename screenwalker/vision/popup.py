"""Popup detection and automatic dismissal.

:class:`PopupHandler` maintains a registry of known popup patterns loaded from
a YAML config file.  Before each step the engine calls
:meth:`PopupHandler.detect_and_handle` to dismiss any overlay dialogs so
they do not block the main automation flow.

Example config (``config/popups.yaml``)::

    popups:
      - id: error_dialog
        indicators: ["Error", "Something went wrong"]
        action: click_button
        button_text: ["OK", "Close"]

      - id: loading_screen
        indicators: ["Loading", "Please wait"]
        action: wait
        wait_timeout: 30
        wait_until_gone: true
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog
import yaml
from PIL import Image

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PopupDefinition:
    """Describes a popup pattern the handler can detect and dismiss.

    Attributes:
        id: Unique identifier used in log messages.
        indicators: OCR text strings that signal the popup is visible.
            Confidence = matched_count / total_indicators.
        templates: Optional template image names for additional visual matching.
            Each matched template adds +0.3 to confidence (capped at 1.0).
        action: Dismissal strategy — ``"click_button"`` | ``"wait"`` | ``"close"``.
        button_text: Ordered list of button labels to try clicking
            (``click_button`` action).
        wait_timeout: Maximum seconds to wait (``wait`` action).
        wait_until_gone: If True, poll until the popup is no longer detected.
        priority: Popups are checked in descending priority order.
        confidence_threshold: Minimum fraction of indicators that must match
            for the popup to be considered detected.
    """

    id: str
    indicators: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    action: str = "click_button"
    button_text: list[str] = field(default_factory=list)
    wait_timeout: float = 30.0
    wait_until_gone: bool = False
    priority: int = 0
    confidence_threshold: float = 0.4


@dataclass(frozen=True)
class PopupMatch:
    """Result of a successful popup detection.

    Attributes:
        definition: The matching popup definition.
        confidence: Detection confidence in ``[0.0, 1.0]``.
        matched_texts: Indicator texts that were found on screen.
    """

    definition: PopupDefinition
    confidence: float
    matched_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PopupHandler
# ---------------------------------------------------------------------------


class PopupHandler:
    """Detects and dismisses popup dialogs before step execution.

    Attributes:
        definitions: Sorted list of popup definitions (highest priority first).
    """

    def __init__(
        self,
        definitions: list[PopupDefinition],
        ocr_engine: Any,
        mouse: Any,
        template_matcher: Any = None,
    ) -> None:
        """Initialise with a list of popup definitions.

        Args:
            definitions: Popup patterns to watch for.
            ocr_engine: OCR engine instance (must implement ``recognize`` and
                ``find_text``).
            mouse: Mouse controller (must implement ``click(x, y)``).
            template_matcher: Optional template matcher instance.
        """
        self.definitions: list[PopupDefinition] = sorted(
            definitions, key=lambda d: d.priority, reverse=True
        )
        self._ocr = ocr_engine
        self._mouse = mouse
        self._matcher = template_matcher

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        ocr_engine: Any,
        mouse: Any,
        template_matcher: Any = None,
    ) -> "PopupHandler":
        """Load popup definitions from a YAML file.

        Args:
            path: Path to a YAML file with a top-level ``popups:`` list.
            ocr_engine: OCR engine instance.
            mouse: Mouse controller instance.
            template_matcher: Optional template matcher.

        Returns:
            Configured :class:`PopupHandler`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Popup config not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        definitions: list[PopupDefinition] = []
        for entry in raw.get("popups", []):
            if not isinstance(entry, dict):
                continue
            try:
                definitions.append(PopupDefinition(
                    id=str(entry["id"]),
                    indicators=list(entry.get("indicators", [])),
                    templates=list(entry.get("templates", [])),
                    action=str(entry.get("action", "click_button")),
                    button_text=list(entry.get("button_text", [])),
                    wait_timeout=float(entry.get("wait_timeout", 30.0)),
                    wait_until_gone=bool(entry.get("wait_until_gone", False)),
                    priority=int(entry.get("priority", 0)),
                    confidence_threshold=float(entry.get("confidence_threshold", 0.4)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("popup_handler.invalid_definition", error=str(exc))

        return cls(definitions, ocr_engine, mouse, template_matcher)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, screenshot: Image.Image) -> PopupMatch | None:
        """Detect whether any known popup is currently visible.

        Definitions are checked in descending priority order.  The first
        definition whose confidence meets :attr:`~PopupDefinition.confidence_threshold`
        is returned.

        Args:
            screenshot: Current screen image.

        Returns:
            :class:`PopupMatch` for the best match, or ``None`` if no popup is
            detected.
        """
        ocr_texts = self._ocr_texts(screenshot)

        for defn in self.definitions:
            confidence, matched = self._score(defn, screenshot, ocr_texts)
            if confidence >= defn.confidence_threshold:
                log.debug(
                    "popup_handler.detected",
                    popup_id=defn.id,
                    confidence=round(confidence, 3),
                    matched=matched,
                )
                return PopupMatch(
                    definition=defn,
                    confidence=confidence,
                    matched_texts=matched,
                )

        return None

    def handle(
        self,
        screenshot: Image.Image,
        match: PopupMatch,
        capture_fn: Callable[[], Image.Image] | None = None,
    ) -> bool:
        """Dismiss a detected popup.

        Args:
            screenshot: Current screen image (used for button location).
            match: Popup match returned by :meth:`detect`.
            capture_fn: Optional callable that returns a fresh screenshot.
                Required for ``wait_until_gone`` behaviour.

        Returns:
            ``True`` if the popup was successfully dismissed.
        """
        defn = match.definition
        log.info("popup_handler.handling", popup_id=defn.id, action=defn.action)

        if defn.action == "click_button":
            return self._handle_click_button(screenshot, defn)
        if defn.action == "close":
            return self._handle_close()
        if defn.action == "wait":
            return self._handle_wait(defn, capture_fn)

        log.warning("popup_handler.unknown_action", action=defn.action)
        return False

    def detect_and_handle(
        self,
        screenshot: Image.Image,
        capture_fn: Callable[[], Image.Image] | None = None,
    ) -> bool:
        """Detect and dismiss a popup in a single call.

        Args:
            screenshot: Current screen image.
            capture_fn: Optional callable for fresh screenshots (needed by the
                ``wait_until_gone`` strategy).

        Returns:
            ``True`` if a popup was found and handled, ``False`` if the screen
            is clear.
        """
        match = self.detect(screenshot)
        if match is None:
            return False
        return self.handle(screenshot, match, capture_fn=capture_fn)

    # ------------------------------------------------------------------
    # Dismissal strategies
    # ------------------------------------------------------------------

    def _handle_click_button(self, screenshot: Image.Image, defn: PopupDefinition) -> bool:
        """Find and click the first available button text."""
        if self._ocr is None:
            log.warning("popup_handler.no_ocr_engine")
            return False

        for btn_text in defn.button_text:
            try:
                result = self._ocr.find_text(screenshot, btn_text, threshold=0.50)
                if result is not None:
                    cx, cy = result.center
                    self._mouse.click(cx, cy)
                    log.info("popup_handler.button_clicked", button=btn_text, popup_id=defn.id)
                    time.sleep(0.3)
                    return True
            except Exception as exc:
                log.debug("popup_handler.button_not_found", button=btn_text, error=str(exc))

        log.warning(
            "popup_handler.no_button_found",
            popup_id=defn.id,
            tried=defn.button_text,
        )
        return False

    def _handle_close(self) -> bool:
        """Press Escape to dismiss the popup."""
        try:
            import pyautogui
            pyautogui.press("escape")
            time.sleep(0.3)
            return True
        except Exception as exc:
            log.warning("popup_handler.escape_failed", error=str(exc))
            return False

    def _handle_wait(
        self,
        defn: PopupDefinition,
        capture_fn: Callable[[], Image.Image] | None,
    ) -> bool:
        """Wait for the popup to disappear or the timeout to expire."""
        if not defn.wait_until_gone or capture_fn is None:
            # Simple time-based wait
            time.sleep(min(defn.wait_timeout, 5.0))
            return True

        deadline = time.monotonic() + defn.wait_timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            fresh = capture_fn()
            if self.detect(fresh) is None:
                log.info("popup_handler.wait_resolved", popup_id=defn.id)
                return True

        log.warning(
            "popup_handler.wait_timeout",
            popup_id=defn.id,
            timeout=defn.wait_timeout,
        )
        return False

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _ocr_texts(self, screenshot: Image.Image) -> list[str]:
        """Run OCR and return a flat list of recognised strings."""
        if self._ocr is None:
            return []
        try:
            results = self._ocr.recognize(screenshot)
            words = [r.text for r in results if r.text.strip()]
            if words:
                words.append(" ".join(words))  # full concatenation for phrase matching
            return words
        except Exception:
            return []

    def _score(
        self,
        defn: PopupDefinition,
        screenshot: Image.Image,
        ocr_texts: list[str],
    ) -> tuple[float, list[str]]:
        """Score *defn* against the current screen content.

        Returns:
            ``(confidence, matched_indicators)``
        """
        try:
            from rapidfuzz import fuzz

            def _found(query: str) -> bool:
                return any(fuzz.partial_ratio(query.lower(), t.lower()) >= 70 for t in ocr_texts)
        except ImportError:
            def _found(query: str) -> bool:
                return any(query.lower() in t.lower() for t in ocr_texts)

        matched: list[str] = []
        for indicator in defn.indicators:
            if _found(indicator):
                matched.append(indicator)

        if not defn.indicators:
            base_confidence = 0.0
        else:
            base_confidence = len(matched) / len(defn.indicators)

        # Template-matching boost
        if self._matcher is not None and defn.templates:
            for tmpl in defn.templates:
                try:
                    if self._matcher.find_one(screenshot, tmpl, threshold=0.75) is not None:
                        base_confidence = min(1.0, base_confidence + 0.3)
                        break
                except Exception:
                    pass

        return base_confidence, matched
