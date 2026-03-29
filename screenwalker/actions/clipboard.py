"""Утилиты чтения/записи буфера обмена.

Использует ``pyperclip`` для кросс-платформенного доступа к буферу обмена и
``pyautogui`` для горячих клавиш копирования/вставки.

Реализует :class:`~screenwalker.actions.protocols.ClipboardProtocol`, что позволяет
заменять менеджер реализацией на базе RDP.
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
    """Кросс-платформенный менеджер чтения/записи буфера обмена.

    Attributes:
        settle_delay: Секунды ожидания после срабатывания горячей клавиши копирования
            перед чтением буфера (даёт ОС время завершить операцию).
    """

    def __init__(self, settle_delay: float = 0.2) -> None:
        """Инициализация ClipboardManager.

        Args:
            settle_delay: Секунды ожидания после инициации копирования перед чтением.
        """
        self.settle_delay = settle_delay

    # ------------------------------------------------------------------
    # Низкоуровневое чтение / запись
    # ------------------------------------------------------------------

    def read(self) -> str:
        """Читает текущее содержимое буфера обмена как строку.

        Returns:
            Текст из буфера обмена или пустая строка, если буфер пуст
            или не содержит текстовых данных.
        """
        result = pyperclip.paste()
        return result if result is not None else ""

    def write(self, text: str) -> None:
        """Записывает *text* в буфер обмена.

        Args:
            text: Строка для размещения в буфере обмена.
        """
        logger.debug("Writing to clipboard", length=len(text))
        pyperclip.copy(text)

    # ------------------------------------------------------------------
    # Вспомогательные методы копирования с последующим чтением
    # ------------------------------------------------------------------

    def copy_selected(self, select_all: bool = False) -> str:
        """Копирует текущее выделение на экране в буфер обмена и возвращает его.

        При необходимости сначала отправляет Ctrl+A для выделения всего текста
        в активном элементе, затем Ctrl+C, ожидает :attr:`settle_delay` и читает результат.

        Args:
            select_all: Если *True*, отправить Ctrl+A перед Ctrl+C.

        Returns:
            Текст, скопированный с экрана.
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
        """Кликает в *(x, y)*, опционально выделяет всё, копирует и читает буфер.

        Последовательность: клик → (опциональный Ctrl+A) → Ctrl+C → ожидание → чтение.

        Args:
            x: Горизонтальная координата экрана для клика.
            y: Вертикальная координата экрана для клика.
            select_all: Если *True*, отправить Ctrl+A после клика.

        Returns:
            Текст, скопированный из элемента в *(x, y)*.
        """
        pyautogui.click(x, y)
        return self.copy_selected(select_all=select_all)

    # ------------------------------------------------------------------
    # Вспомогательный метод вставки
    # ------------------------------------------------------------------

    def paste(self) -> None:
        """Вставляет содержимое буфера обмена в текущую позицию курсора через Ctrl+V."""
        if _PLATFORM == "Darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

    def clear(self) -> None:
        """Очищает буфер обмена записью пустой строки."""
        self.write("")

    # ------------------------------------------------------------------
    # Устаревший псевдоним
    # ------------------------------------------------------------------

    def read_selection(self) -> str:
        """Псевдоним для :meth:`copy_selected` (без select-all).

        Устарел: используйте :meth:`copy_selected` напрямую.
        """
        return self.copy_selected(select_all=False)
