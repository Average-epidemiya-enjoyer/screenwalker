"""Keyboard action primitives.

Wraps PyAutoGUI to provide cross-platform text input and hotkey support.
Cyrillic and other non-ASCII text is routed through the clipboard to avoid
PyAutoGUI's ASCII-only ``typewrite`` limitation.

Implements :class:`~screenwalker.actions.protocols.KeyboardInputProtocol` so
the controller can be replaced with an RDP-backed implementation without
changing call sites.
"""

from __future__ import annotations

import platform

import pyautogui
import pyperclip
import structlog

logger = structlog.get_logger(__name__)

# Mapping from common alias names to PyAutoGUI key names
_KEY_ALIASES: dict[str, str] = {
    "return": "enter",
    "esc": "escape",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "win": "winleft",
    "cmd": "command",
    "opt": "option",
}

_PLATFORM = platform.system()


def _normalise_key(key: str) -> str:
    """Map an alias to the canonical PyAutoGUI key name.

    Args:
        key: Raw key name from the scenario YAML.

    Returns:
        Canonical PyAutoGUI key string.
    """
    return _KEY_ALIASES.get(key.lower(), key.lower())


def _has_non_ascii(text: str) -> bool:
    """Return True if *text* contains any character outside printable ASCII."""
    return any(ord(c) > 127 for c in text)


class KeyboardController:
    """PyAutoGUI-backed keyboard controller with Cyrillic support.

    Cyrillic (and any non-ASCII) text is typed via clipboard paste so that
    platform input method restrictions in ``pyautogui.typewrite`` are bypassed.

    Attributes:
        typing_interval: Delay between keystrokes when typing (seconds).
    """

    def __init__(self, typing_interval: float = 0.03) -> None:
        """Initialise KeyboardController.

        Args:
            typing_interval: Seconds between individual key presses when typing.
        """
        self.typing_interval = typing_interval

    # ------------------------------------------------------------------
    # Text input
    # ------------------------------------------------------------------

    def type_text(self, text: str, interval: float | None = None) -> None:
        """Type *text* character by character.

        Non-ASCII characters (Cyrillic, CJK, etc.) are inserted via
        :meth:`type_unicode` instead of ``pyautogui.typewrite``.

        Args:
            text: Text to type. Supports printable ASCII and Unicode.
            interval: Override typing interval for this call.
        """
        iv = interval if interval is not None else self.typing_interval
        logger.debug("Type text", length=len(text), has_unicode=_has_non_ascii(text))
        if _has_non_ascii(text):
            self.type_unicode(text)
        else:
            pyautogui.typewrite(text, interval=iv)

    def type_unicode(self, text: str) -> None:
        """Insert Unicode text via clipboard paste (Ctrl+V / Cmd+V).

        Writes *text* to the system clipboard and then sends the platform
        paste shortcut.  This is the only reliable way to input non-ASCII
        characters through PyAutoGUI.

        Args:
            text: Unicode string to insert at the current cursor position.
        """
        logger.debug("Type unicode via clipboard", length=len(text))
        pyperclip.copy(text)
        if _PLATFORM == "Darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

    # ------------------------------------------------------------------
    # Hotkeys / single keys
    # ------------------------------------------------------------------

    def hotkey(self, *keys: str) -> None:
        """Press a key combination simultaneously.

        Args:
            *keys: Key names to press together (e.g. ``"ctrl", "s"``).
                Keys are normalised via :func:`_normalise_key`.

        Example:
            >>> kb = KeyboardController()
            >>> kb.hotkey("ctrl", "alt", "del")
        """
        normalised = [_normalise_key(k) for k in keys]
        logger.debug("Hotkey", keys=normalised)
        pyautogui.hotkey(*normalised)

    def press(self, key: str) -> None:
        """Press and release a single key.

        Args:
            key: Key name (will be normalised via :func:`_normalise_key`).
        """
        normalised = _normalise_key(key)
        logger.debug("Key press", key=normalised)
        pyautogui.press(normalised)

    def key_down(self, key: str) -> None:
        """Hold a key down without releasing.

        Args:
            key: Key name (will be normalised).
        """
        normalised = _normalise_key(key)
        logger.debug("Key down", key=normalised)
        pyautogui.keyDown(normalised)

    def key_up(self, key: str) -> None:
        """Release a previously held key.

        Args:
            key: Key name (will be normalised).
        """
        normalised = _normalise_key(key)
        logger.debug("Key up", key=normalised)
        pyautogui.keyUp(normalised)

    # ------------------------------------------------------------------
    # Compound helpers
    # ------------------------------------------------------------------

    def select_all(self) -> None:
        """Send Ctrl+A (or Cmd+A on macOS) to select all text."""
        if _PLATFORM == "Darwin":
            self.hotkey("command", "a")
        else:
            self.hotkey("ctrl", "a")

    def clear_field(self) -> None:
        """Select all text in the focused field and delete it."""
        self.select_all()
        self.press("delete")
