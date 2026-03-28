"""Clipboard read/write utilities.

Uses ``pyperclip`` for cross-platform clipboard access.  Provides an optional
"read via Ctrl+C" pattern that copies the current selection to the clipboard
before reading it.
"""

from __future__ import annotations

import time

import structlog

logger = structlog.get_logger(__name__)


class ClipboardManager:
    """Cross-platform clipboard read/write manager backed by pyperclip.

    Attributes:
        settle_delay: Seconds to wait after triggering Ctrl+C before reading
            the clipboard (allows the OS to complete the copy).
    """

    def __init__(self, settle_delay: float = 0.2) -> None:
        """Initialize ClipboardManager.

        Args:
            settle_delay: Seconds to wait after a copy trigger before reading.
        """
        self.settle_delay = settle_delay

    def read(self) -> str:
        """Read the current clipboard contents as a string.

        Returns:
            Clipboard text. Returns an empty string if the clipboard is empty
            or does not contain text.

        Raises:
            RuntimeError: If the clipboard cannot be accessed on the current platform.
        """
        # TODO: import pyperclip; return pyperclip.paste() or ""
        raise NotImplementedError("TODO: implement ClipboardManager.read")

    def write(self, text: str) -> None:
        """Write *text* to the clipboard.

        Args:
            text: String to place on the clipboard.
        """
        logger.debug("Writing to clipboard", length=len(text))
        # TODO: import pyperclip; pyperclip.copy(text)
        raise NotImplementedError("TODO: implement ClipboardManager.write")

    def read_selection(self) -> str:
        """Copy the current selection to clipboard via Ctrl+C, then read it.

        Sends the platform-appropriate copy shortcut and waits
        :attr:`settle_delay` seconds before reading the clipboard.

        Returns:
            Text that was selected on screen.

        Raises:
            RuntimeError: If clipboard access fails.
        """
        import platform

        from screenwalker.actions.keyboard import KeyboardController

        kb = KeyboardController()
        if platform.system() == "Darwin":
            kb.hotkey("command", "c")
        else:
            kb.hotkey("ctrl", "c")

        time.sleep(self.settle_delay)
        return self.read()

    def paste(self) -> None:
        """Paste clipboard contents at the current cursor position via Ctrl+V."""
        import platform

        from screenwalker.actions.keyboard import KeyboardController

        kb = KeyboardController()
        if platform.system() == "Darwin":
            kb.hotkey("command", "v")
        else:
            kb.hotkey("ctrl", "v")

    def clear(self) -> None:
        """Clear the clipboard by writing an empty string."""
        self.write("")
