"""Протоколы абстракции ввода для будущих RDP/remote-input бэкендов.

Определяют структурные интерфейсы для мыши, клавиатуры и буфера обмена,
чтобы конкретные реализации (PyAutoGUI, RDP, VNC…) можно было менять прозрачно.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MouseInputProtocol(Protocol):
    """Структурный протокол для бэкендов управления мышью."""

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
    """Структурный протокол для бэкендов управления клавиатурой."""

    def type_text(self, text: str, interval: float = 0.05) -> None: ...

    def hotkey(self, *keys: str) -> None: ...

    def press(self, key: str) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...


@runtime_checkable
class ClipboardProtocol(Protocol):
    """Структурный протокол для бэкендов буфера обмена."""

    def read(self) -> str: ...

    def write(self, text: str) -> None: ...

    def copy_selected(self, select_all: bool = False) -> str: ...

    def copy_and_read(self, x: int, y: int, select_all: bool = False) -> str: ...
