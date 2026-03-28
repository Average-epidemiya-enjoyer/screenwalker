"""Screen and window capture utilities.

Provides both a :class:`ScreenCapture` class (recommended) and module-level
convenience functions that delegate to a default instance.

All returned images are ``PIL.Image.Image`` objects in RGB mode.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import pyautogui
import structlog
from PIL import Image

from screenwalker.vision.screen_state import BBox

logger = structlog.get_logger(__name__)


class WindowInfo(NamedTuple):
    """Basic metadata about a captured window.

    Attributes:
        title: Window title string.
        bbox: Position and size of the window on screen.
        pid: OS process ID owning the window (if available).
    """

    title: str
    bbox: BBox
    pid: int | None


class ScreenCapture:
    """Cross-platform screenshot helper with per-step image caching.

    Attributes:
        screenshot_delay: Seconds to pause before each capture (UI settle time).
        debug_dir: Directory for :meth:`save_debug` output.
    """

    def __init__(
        self,
        screenshot_delay: float = 0.3,
        debug_dir: str | Path = "logs",
    ) -> None:
        """Initialise ScreenCapture.

        Args:
            screenshot_delay: Pause before capturing so the UI has time to settle.
            debug_dir: Root directory for debug screenshots written by
                :meth:`save_debug`.
        """
        self.screenshot_delay = screenshot_delay
        self.debug_dir = Path(debug_dir)
        self._last_image: Image.Image | None = None

    # ------------------------------------------------------------------
    # Public capture API
    # ------------------------------------------------------------------

    def capture_full(self) -> Image.Image:
        """Capture the entire primary screen.

        Returns:
            Full-screen screenshot as a PIL Image in RGB mode.
        """
        self._settle()
        img = pyautogui.screenshot()
        img = img.convert("RGB")
        self._last_image = img
        logger.debug("Captured full screen", size=img.size)
        return img

    def capture_region(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """Capture a rectangular region of the screen.

        Args:
            x: Left edge in screen pixels.
            y: Top edge in screen pixels.
            w: Width in pixels.
            h: Height in pixels.

        Returns:
            Cropped screenshot as a PIL Image in RGB mode.
        """
        self._settle()
        img = pyautogui.screenshot(region=(x, y, w, h))
        img = img.convert("RGB")
        self._last_image = img
        logger.debug("Captured region", x=x, y=y, w=w, h=h)
        return img

    def capture_window(self, title: str) -> Image.Image:
        """Capture the window whose title contains *title*.

        Performs a case-insensitive partial-title match and activates the
        window before capturing.

        Args:
            title: Partial or full window title (case-insensitive).

        Returns:
            Window screenshot as a PIL Image in RGB mode.

        Raises:
            ValueError: If no window matching *title* is found.
            RuntimeError: If the required platform library is unavailable.
        """
        system = platform.system()
        if system == "Windows":
            return self._capture_window_windows(title)
        elif system == "Darwin":
            return self._capture_window_macos(title)
        else:
            return self._capture_window_linux(title)

    # ------------------------------------------------------------------
    # Last-image cache
    # ------------------------------------------------------------------

    @property
    def last_image(self) -> Image.Image | None:
        """Most recently captured image, or None if no capture has been made."""
        return self._last_image

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def save_debug(self, image: Image.Image, step_name: str) -> Path:
        """Save *image* to the debug directory with a sanitised filename.

        Args:
            image: PIL Image to persist.
            step_name: Human-readable label used as the filename stem.

        Returns:
            Absolute path of the written file.
        """
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        safe_name = step_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        path = self.debug_dir / f"{safe_name}.png"
        image.save(path, "PNG")
        logger.debug("Saved debug screenshot", path=str(path), step=step_name)
        return path.resolve()

    # ------------------------------------------------------------------
    # Platform-specific window capture
    # ------------------------------------------------------------------

    def _capture_window_windows(self, title: str) -> Image.Image:
        try:
            import pygetwindow as gw  # installed as part of pyautogui on Windows
        except ImportError as exc:
            raise RuntimeError(
                "pygetwindow is required for window capture on Windows. "
                "It is normally installed automatically with pyautogui."
            ) from exc

        windows = gw.getWindowsWithTitle(title)
        if not windows:
            raise ValueError(f"No window found with title matching: {title!r}")

        win = windows[0]
        try:
            win.activate()
            time.sleep(0.2)
        except Exception:
            pass  # activation failure is non-fatal; capture whatever is visible

        x, y, w, h = win.left, win.top, win.width, win.height
        self._settle()
        img = pyautogui.screenshot(region=(x, y, w, h))
        img = img.convert("RGB")
        self._last_image = img
        logger.debug("Captured window (Windows)", title=title, bbox=(x, y, w, h))
        return img

    def _capture_window_macos(self, title: str) -> Image.Image:  # pragma: no cover
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PyObjC Quartz bindings are required for window capture on macOS: "
                "pip install pyobjc-framework-Quartz"
            ) from exc

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        title_lower = title.lower()
        match = next(
            (
                w
                for w in window_list
                if title_lower in (w.get("kCGWindowName") or "").lower()
            ),
            None,
        )
        if match is None:
            raise ValueError(f"No window found with title matching: {title!r}")

        bounds = match["kCGWindowBounds"]
        x = int(bounds["X"])
        y = int(bounds["Y"])
        w = int(bounds["Width"])
        h = int(bounds["Height"])
        return self.capture_region(x, y, w, h)

    def _capture_window_linux(self, title: str) -> Image.Image:  # pragma: no cover
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "xdotool is required for window capture on Linux: "
                "sudo apt install xdotool"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"No window found with title matching: {title!r}"
            ) from exc

        wid = result.stdout.strip().split("\n")[0]
        geom_result = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", wid],
            capture_output=True,
            text=True,
            check=True,
        )
        geom: dict[str, int] = {}
        for line in geom_result.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                geom[k.strip()] = int(v.strip())

        return self.capture_region(geom["X"], geom["Y"], geom["WIDTH"], geom["HEIGHT"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _settle(self) -> None:
        if self.screenshot_delay > 0:
            time.sleep(self.screenshot_delay)


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward compatible)
# ---------------------------------------------------------------------------

_default_capture = ScreenCapture(screenshot_delay=0.0)


def capture_screen(delay: float = 0.0) -> Image.Image:
    """Capture the entire primary screen.

    Args:
        delay: Optional pause in seconds before capturing.

    Returns:
        Full-screen screenshot as a PIL Image in RGB mode.
    """
    _default_capture.screenshot_delay = delay
    return _default_capture.capture_full()


def capture_region(bbox: BBox, delay: float = 0.0) -> Image.Image:
    """Capture a rectangular region of the screen.

    Args:
        bbox: Screen region to capture (x, y, w, h in screen pixels).
        delay: Optional pause before capturing.

    Returns:
        Cropped screenshot as a PIL Image in RGB mode.
    """
    _default_capture.screenshot_delay = delay
    return _default_capture.capture_region(bbox.x, bbox.y, bbox.w, bbox.h)


def capture_window(title: str, delay: float = 0.0) -> Image.Image:
    """Capture the window with the given title.

    Args:
        title: Partial or full window title string (case-insensitive).
        delay: Optional pause before capturing.

    Returns:
        Window screenshot as a PIL Image in RGB mode.

    Raises:
        ValueError: If no window matching *title* is found.
    """
    _default_capture.screenshot_delay = delay
    return _default_capture.capture_window(title)


def list_windows() -> list[WindowInfo]:
    """Return metadata for all visible top-level windows.

    Returns:
        List of :class:`WindowInfo`. On unsupported platforms returns an
        empty list rather than raising.
    """
    system = platform.system()
    if system == "Windows":
        try:
            import pygetwindow as gw

            result: list[WindowInfo] = []
            for win in gw.getAllWindows():
                if win.title:
                    bbox = BBox(x=win.left, y=win.top, w=win.width, h=win.height)
                    result.append(WindowInfo(title=win.title, bbox=bbox, pid=None))
            return result
        except ImportError:
            return []
    return []


def save_image(image: Image.Image, path: Path | str, quality: int = 95) -> Path:
    """Save a PIL image to disk in PNG or JPEG format.

    Args:
        image: Image to save.
        path: Destination file path. Extension determines format.
        quality: JPEG quality (1–95); ignored for PNG.

    Returns:
        Absolute resolved path of the saved file.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(path, "JPEG", quality=quality)
    else:
        image.save(path, "PNG")
    return path
