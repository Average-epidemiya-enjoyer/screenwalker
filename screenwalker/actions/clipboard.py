"""Clipboard read/write utilities.

Uses ``pyperclip`` for cross-platform clipboard access and ``pyautogui`` for
the copy/paste keyboard shortcuts.

Implements :class:`~screenwalker.actions.protocols.ClipboardProtocol` so the
manager can be replaced with an RDP-backed implementation.
"""

from __future__ import annotations

import platform
import time

import pyautogui
import pyperclip
import structlog

logger = structlog.get_logger(__name__)

_PLATFORM = platform.system()


class ClipboardManager:
    """Cross-platform clipboard read/write manager.

    Attributes:
        settle_delay: Seconds to wait after triggering a copy shortcut before
            reading the clipboard (allows the OS to complete the operation).
    """

    def __init__(self, settle_delay: float = 0.2) -> None:
        """Initialise ClipboardManager.

        Args:
            settle_delay: Seconds to wait after a copy trigger before reading.
        """
        self.settle_delay = settle_delay

    # ------------------------------------------------------------------
    # Low-level read / write
    # ------------------------------------------------------------------

    def read(self) -> str:
        """Read the current clipboard contents as a string.

        Returns:
            Clipboard text, or an empty string if the clipboard is empty or
            does not contain text data.
        """
        result = pyperclip.paste()
        return result if result is not None else ""

    def write(self, text: str) -> None:
        """Write *text* to the clipboard.

        Args:
            text: String to place on the clipboard.
        """
        logger.debug("Writing to clipboard", length=len(text))
        pyperclip.copy(text)

    # ------------------------------------------------------------------
    # Copy-then-read helpers
    # ------------------------------------------------------------------

    def copy_selected(self, select_all: bool = False) -> str:
        """Copy the current on-screen selection to the clipboard and read it.

        Optionally sends Ctrl+A first to select all text in the focused
        element, then Ctrl+C, waits :attr:`settle_delay`, and reads the result.

        Args:
            select_all: If *True*, send Ctrl+A before Ctrl+C.

        Returns:
            Text that was copied from the screen.
        """
        if select_all:
            if _PLATFORM == "Darwin":
                pyautogui.hotkey("command", "a")
            else:
                pyautogui.hotkey("ctrl", "a")

        if _PLATFORM == "Darwin":
            pyautogui.hotkey("command", "c")
        else:
            pyautogui.hotkey("ctrl", "c")

        time.sleep(self.settle_delay)
        text = self.read()
        logger.debug("Copied selection", select_all=select_all, length=len(text))
        return text

    def copy_and_read(
        self,
        x: int,
        y: int,
        select_all: bool = False,
    ) -> str:
        """Click at *(x, y)*, optionally select all, copy, and read the clipboard.

        Sequence: click → (optional Ctrl+A) → Ctrl+C → wait → read.

        Args:
            x: Horizontal screen coordinate to click.
            y: Vertical screen coordinate to click.
            select_all: If *True*, send Ctrl+A after clicking.

        Returns:
            Text copied from the element at *(x, y)*.
        """
        pyautogui.click(x, y)
        return self.copy_selected(select_all=select_all)

    # ------------------------------------------------------------------
    # Paste helper
    # ------------------------------------------------------------------

    def paste(self) -> None:
        """Paste clipboard contents at the current cursor position via Ctrl+V."""
        if _PLATFORM == "Darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

    def clear(self) -> None:
        """Clear the clipboard by writing an empty string."""
        self.write("")

    # ------------------------------------------------------------------
    # Backward-compatible alias
    # ------------------------------------------------------------------

    def read_selection(self) -> str:
        """Alias for :meth:`copy_selected` (no select-all).

        Deprecated: use :meth:`copy_selected` directly.
        """
        return self.copy_selected(select_all=False)
