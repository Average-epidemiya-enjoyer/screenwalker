"""Keyboard action primitives.

Wraps PyAutoGUI to provide cross-platform text input and hotkey support.
Key names follow the PyAutoGUI convention (e.g. ``"enter"``, ``"ctrl"``,
``"alt"``, ``"F4"``).
"""

from __future__ import annotations

import time

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


def _normalise_key(key: str) -> str:
    """Map an alias to the canonical PyAutoGUI key name.

    Args:
        key: Raw key name from the scenario YAML.

    Returns:
        Canonical PyAutoGUI key string.
    """
    return _KEY_ALIASES.get(key.lower(), key.lower())


class KeyboardController:
    """Stateless keyboard controller backed by PyAutoGUI.

    Attributes:
        typing_interval: Delay between keystrokes when typing (seconds).
    """

    def __init__(self, typing_interval: float = 0.03) -> None:
        """Initialize KeyboardController.

        Args:
            typing_interval: Seconds between individual key presses when typing.
        """
        self.typing_interval = typing_interval

    def type_text(self, text: str, interval: float | None = None) -> None:
        """Type a string character by character.

        Args:
            text: Text to type. Supports printable ASCII and Unicode.
            interval: Override typing interval for this call.
        """
        iv = interval if interval is not None else self.typing_interval
        logger.debug("Type text", length=len(text))
        # TODO: pyautogui.typewrite(text, interval=iv)
        #       Note: typewrite doesn't support Unicode — for non-ASCII,
        #       use pyautogui.hotkey('ctrl', 'a') then clipboard paste.
        raise NotImplementedError("TODO: implement type_text")

    def type_unicode(self, text: str) -> None:
        """Type Unicode text via clipboard paste.

        Falls back to clipboard paste for characters not supported by
        ``pyautogui.typewrite``.

        Args:
            text: Unicode string to insert.
        """
        # TODO: copy text to clipboard, then Ctrl+V
        raise NotImplementedError("TODO: implement type_unicode")

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
        # TODO: pyautogui.hotkey(*normalised)
        raise NotImplementedError("TODO: implement hotkey")

    def press(self, key: str) -> None:
        """Press and release a single key.

        Args:
            key: Key name (will be normalised).
        """
        normalised = _normalise_key(key)
        logger.debug("Key press", key=normalised)
        # TODO: pyautogui.press(normalised)
        raise NotImplementedError("TODO: implement press")

    def key_down(self, key: str) -> None:
        """Hold a key down without releasing.

        Args:
            key: Key name (will be normalised).
        """
        # TODO: pyautogui.keyDown(_normalise_key(key))
        raise NotImplementedError("TODO: implement key_down")

    def key_up(self, key: str) -> None:
        """Release a previously held key.

        Args:
            key: Key name (will be normalised).
        """
        # TODO: pyautogui.keyUp(_normalise_key(key))
        raise NotImplementedError("TODO: implement key_up")

    def select_all(self) -> None:
        """Send Ctrl+A (or Cmd+A on macOS) to select all text."""
        import platform

        if platform.system() == "Darwin":
            self.hotkey("command", "a")
        else:
            self.hotkey("ctrl", "a")

    def clear_field(self) -> None:
        """Select all and delete — useful before typing into a text field."""
        self.select_all()
        self.press("delete")
