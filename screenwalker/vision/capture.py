"""Screen and window capture utilities.

Wraps PyAutoGUI and PIL to provide cross-platform screenshot primitives.
All returned images are ``PIL.Image.Image`` objects in RGB mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

from PIL import Image

from screenwalker.vision.screen_state import BBox


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


def capture_screen(delay: float = 0.0) -> Image.Image:
    """Capture the entire primary screen.

    Args:
        delay: Optional pause in seconds before capturing (allows UI to settle).

    Returns:
        Full-screen screenshot as a PIL Image in RGB mode.

    Example:
        >>> img = capture_screen(delay=0.3)
        >>> img.size  # (width, height)
        (1920, 1080)
    """
    if delay > 0:
        time.sleep(delay)
    # TODO: import pyautogui; return ImageGrab.grab() or pyautogui.screenshot()
    raise NotImplementedError("TODO: implement capture_screen using pyautogui/PIL")


def capture_region(bbox: BBox, delay: float = 0.0) -> Image.Image:
    """Capture a rectangular region of the screen.

    Args:
        bbox: Screen region to capture (x, y, w, h in screen pixels).
        delay: Optional pause before capturing.

    Returns:
        Cropped screenshot as a PIL Image in RGB mode.
    """
    if delay > 0:
        time.sleep(delay)
    # TODO: capture full screen then crop to bbox
    raise NotImplementedError("TODO: implement capture_region")


def capture_window(title: str, delay: float = 0.0) -> Image.Image:
    """Capture the window with the given title.

    Locates the window by partial title match on all platforms and captures
    only its client area.

    Args:
        title: Partial or full window title string (case-insensitive).
        delay: Optional pause before capturing.

    Returns:
        Window screenshot as a PIL Image in RGB mode.

    Raises:
        ValueError: If no window matching *title* is found.
    """
    if delay > 0:
        time.sleep(delay)
    # TODO: platform-specific window locator (pygetwindow on Win/Mac, xdotool on Linux)
    raise NotImplementedError("TODO: implement capture_window")


def list_windows() -> list[WindowInfo]:
    """Return metadata for all visible top-level windows.

    Returns:
        List of :class:`WindowInfo` sorted by window z-order (front first).
    """
    # TODO: platform-specific window enumeration
    raise NotImplementedError("TODO: implement list_windows")


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
