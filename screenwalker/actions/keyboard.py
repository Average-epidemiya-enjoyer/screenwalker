"""Примитивы действий клавиатуры.

Обёртка над PyAutoGUI для кросс-платформенного ввода текста и поддержки горячих клавиш.
Кириллица и другой не-ASCII текст направляется через буфер обмена для обхода
ограничения PyAutoGUI (``typewrite`` работает только с ASCII).

Реализует :class:`~screenwalker.actions.protocols.KeyboardInputProtocol`, что позволяет
заменять контроллер реализацией на базе RDP без изменения вызывающего кода.
"""

from __future__ import annotations

import platform

import pyautogui
import pyperclip
import structlog

logger = structlog.get_logger(__name__)

# Сопоставление псевдонимов с именами клавиш PyAutoGUI
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
    """Сопоставляет псевдоним с каноническим именем клавиши PyAutoGUI.

    Args:
        key: Сырое имя клавиши из YAML-сценария.

    Returns:
        Каноническое имя клавиши PyAutoGUI.
    """
    return _KEY_ALIASES.get(key.lower(), key.lower())


def _has_non_ascii(text: str) -> bool:
    """Возвращает True, если *text* содержит символы вне печатного ASCII."""
    return any(ord(c) > 127 for c in text)


class KeyboardController:
    """Контроллер клавиатуры на базе PyAutoGUI с поддержкой кириллицы.

    Кириллический (и любой не-ASCII) текст вводится через вставку из буфера обмена,
    чтобы обойти ограничения платформенных методов ввода в ``pyautogui.typewrite``.

    Attributes:
        typing_interval: Задержка между нажатиями клавиш при вводе (секунды).
    """

    def __init__(self, typing_interval: float = 0.03) -> None:
        """Инициализация KeyboardController.

        Args:
            typing_interval: Задержка между отдельными нажатиями клавиш при вводе текста.
        """
        self.typing_interval = typing_interval

    # ------------------------------------------------------------------
    # Ввод текста
    # ------------------------------------------------------------------

    def type_text(self, text: str, interval: float | None = None) -> None:
        """Вводит *text* посимвольно.

        Не-ASCII символы (кириллица, CJK и т.д.) вставляются через
        :meth:`type_unicode` вместо ``pyautogui.typewrite``.

        Args:
            text: Текст для ввода. Поддерживает печатный ASCII и Unicode.
            interval: Переопределяет интервал ввода для этого вызова.
        """
        iv = interval if interval is not None else self.typing_interval
        logger.debug("Type text", length=len(text), has_unicode=_has_non_ascii(text))
        if _has_non_ascii(text):
            self.type_unicode(text)
        else:
            pyautogui.typewrite(text, interval=iv)

    def type_unicode(self, text: str) -> None:
        """Вставляет Unicode-текст через буфер обмена (Ctrl+V / Cmd+V).

        Записывает *text* в системный буфер обмена и затем отправляет
        платформенное сочетание вставки. Это единственный надёжный способ
        ввода не-ASCII символов через PyAutoGUI.

        Args:
            text: Unicode-строка для вставки в текущую позицию курсора.
        """
        logger.debug("Type unicode via clipboard", length=len(text))
        pyperclip.copy(text)
        if _PLATFORM == "Darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

    # ------------------------------------------------------------------
    # Горячие клавиши / одиночные клавиши
    # ------------------------------------------------------------------

    def hotkey(self, *keys: str) -> None:
        """Нажимает комбинацию клавиш одновременно.

        Args:
            *keys: Имена клавиш для одновременного нажатия (например, ``"ctrl", "s"``).
                Клавиши нормализуются через :func:`_normalise_key`.

        Example:
            >>> kb = KeyboardController()
            >>> kb.hotkey("ctrl", "alt", "del")
        """
        normalised = [_normalise_key(k) for k in keys]
        logger.debug("Hotkey", keys=normalised)
        pyautogui.hotkey(*normalised)

    def press(self, key: str) -> None:
        """Нажимает и отпускает одну клавишу.

        Args:
            key: Имя клавиши (будет нормализовано через :func:`_normalise_key`).
        """
        normalised = _normalise_key(key)
        logger.debug("Key press", key=normalised)
        pyautogui.press(normalised)

    def key_down(self, key: str) -> None:
        """Удерживает клавишу нажатой без отпускания.

        Args:
            key: Имя клавиши (будет нормализовано).
        """
        normalised = _normalise_key(key)
        logger.debug("Key down", key=normalised)
        pyautogui.keyDown(normalised)

    def key_up(self, key: str) -> None:
        """Отпускает ранее удерживаемую клавишу.

        Args:
            key: Имя клавиши (будет нормализовано).
        """
        normalised = _normalise_key(key)
        logger.debug("Key up", key=normalised)
        pyautogui.keyUp(normalised)

    # ------------------------------------------------------------------
    # Составные вспомогательные методы
    # ------------------------------------------------------------------

    def select_all(self) -> None:
        """Отправляет Ctrl+A (или Cmd+A на macOS) для выделения всего текста."""
        if _PLATFORM == "Darwin":
            self.hotkey("command", "a")
        else:
            self.hotkey("ctrl", "a")

    def clear_field(self) -> None:
        """Выделяет весь текст в активном поле и удаляет его."""
        self.select_all()
        self.press("delete")
