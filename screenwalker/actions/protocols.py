"""Input abstraction protocols for future RDP/remote-input backends.

Define structural interfaces for mouse, keyboard, and clipboard so concrete
implementations (PyAutoGUI, RDP, VNC…) can be swapped transparently.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MouseInputProtocol(Protocol):
    """Structural protocol for mouse input backends."""

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None: ...

    def double_click(self, x: int, y: int) -> None: ...

    def right_click(self, x: int, y: int) -> None: ...

    def move_to(self, x: int, y: int, duration: float = 0.3) -> None: ...

    def scroll(
        self, x: int, y: int, clicks: int = 3, direction: str = "down"
    ) -> None: ...

    def drag_to(
        self, start_x: int, start_y: int, end_x: int, end_y: int
    ) -> None: ...


@runtime_checkable
class KeyboardInputProtocol(Protocol):
    """Structural protocol for keyboard input backends."""

    def type_text(self, text: str, interval: float = 0.05) -> None: ...

    def hotkey(self, *keys: str) -> None: ...

    def press(self, key: str) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...


@runtime_checkable
class ClipboardProtocol(Protocol):
    """Structural protocol for clipboard backends."""

    def read(self) -> str: ...

    def write(self, text: str) -> None: ...

    def copy_selected(self, select_all: bool = False) -> str: ...

    def copy_and_read(self, x: int, y: int, select_all: bool = False) -> str: ...
