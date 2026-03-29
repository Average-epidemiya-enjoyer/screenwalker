"""Примитивы действий мыши.

Обёртка над PyAutoGUI для типизированных кросс-платформенных операций мыши
с опциональным человекоподобным джиттером (случайное смещение координат + задержка перед действием).

Реализует :class:`~screenwalker.actions.protocols.MouseInputProtocol`, что позволяет
заменять контроллер реализацией на базе RDP или VNC без изменения вызывающего кода.
"""

from __future__ import annotations

import random
import time

import pyautogui
import structlog

from screenwalker.vision.screen_state import BBox

logger = structlog.get_logger(__name__)


class MouseController:
    """Контроллер мыши на базе PyAutoGUI с опциональным очеловечиванием.

    Attributes:
        move_duration: Длительность анимации движения мыши по умолчанию (секунды).
        failsafe: Активен ли угловой failsafe PyAutoGUI.
        humanize: При значении *True* добавляет случайный джиттер ±*humanize_offset_px*
            к координатам и случайную задержку перед действием от
            *humanize_delay_min_ms* до *humanize_delay_max_ms* миллисекунд.
        humanize_offset_px: Максимальное случайное смещение (±) в пикселях по каждой оси.
        humanize_delay_min_ms: Минимальная случайная задержка перед действием (мс).
        humanize_delay_max_ms: Максимальная случайная задержка перед действием (мс).
    """

    def __init__(
        self,
        move_duration: float = 0.2,
        failsafe: bool = True,
        humanize: bool = True,
        humanize_offset_px: int = 2,
        humanize_delay_min_ms: int = 50,
        humanize_delay_max_ms: int = 150,
    ) -> None:
        """Инициализация MouseController.

        Args:
            move_duration: Длительность плавного движения мыши в секундах.
                Укажите 0 для мгновенного перемещения.
            failsafe: Включить угловой failsafe PyAutoGUI.
            humanize: Включить случайный джиттер для всех действий.
            humanize_offset_px: Величина джиттера (±) в пикселях по каждой оси.
            humanize_delay_min_ms: Нижняя граница случайной задержки.
            humanize_delay_max_ms: Верхняя граница случайной задержки.
        """
        self.move_duration = move_duration
        self.failsafe = failsafe
        self.humanize = humanize
        self.humanize_offset_px = humanize_offset_px
        self.humanize_delay_min_ms = humanize_delay_min_ms
        self.humanize_delay_max_ms = humanize_delay_max_ms
        pyautogui.FAILSAFE = failsafe

    # ------------------------------------------------------------------
    # Действия клика
    # ------------------------------------------------------------------

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Перемещается в *(x, y)* и выполняет клик.

        Args:
            x: Горизонтальная координата экрана.
            y: Вертикальная координата экрана.
            button: Кнопка мыши — ``"left"``, ``"right"`` или ``"middle"``.
            clicks: Количество кликов.
        """
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Mouse click", x=ax, y=ay, button=button, clicks=clicks)
        pyautogui.click(
            ax,
            ay,
            clicks=clicks,
            button=button,
            duration=self.move_duration,
        )

    def click_bbox(self, bbox: BBox, button: str = "left") -> None:
        """Кликает по центру ограничивающего прямоугольника.

        Args:
            bbox: Целевой ограничивающий прямоугольник.
            button: Кнопка мыши для клика.
        """
        cx, cy = bbox.center
        self.click(cx, cy, button=button)

    def double_click(self, x: int, y: int) -> None:
        """Двойной клик в *(x, y)*.

        Args:
            x: Горизонтальная координата экрана.
            y: Вертикальная координата экрана.
        """
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Double click", x=ax, y=ay)
        pyautogui.doubleClick(ax, ay, duration=self.move_duration)

    def right_click(self, x: int, y: int) -> None:
        """Правый клик в *(x, y)*.

        Args:
            x: Горизонтальная координата экрана.
            y: Вертикальная координата экрана.
        """
        self.click(x, y, button="right")

    # ------------------------------------------------------------------
    # Перемещение
    # ------------------------------------------------------------------

    def move_to(self, x: int, y: int, duration: float | None = None) -> None:
        """Перемещает курсор мыши в *(x, y)* без клика.

        Args:
            x: Целевая горизонтальная координата.
            y: Целевая вертикальная координата.
            duration: Длительность анимации в секундах. По умолчанию
                :attr:`move_duration`.
        """
        d = duration if duration is not None else self.move_duration
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Move to", x=ax, y=ay, duration=d)
        pyautogui.moveTo(ax, ay, duration=d)

    # ------------------------------------------------------------------
    # Прокрутка
    # ------------------------------------------------------------------

    def scroll(
        self,
        x: int,
        y: int,
        clicks: int = 3,
        direction: str = "down",
    ) -> None:
        """Прокручивает колёсико мыши в *(x, y)*.

        Args:
            x: Горизонтальная координата экрана.
            y: Вертикальная координата экрана.
            clicks: Количество делений прокрутки.
            direction: ``"up"`` (вверх) или ``"down"`` (вниз).

        Raises:
            ValueError: Если *direction* не равно ``"up"`` или ``"down"``.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got: {direction!r}")
        amount = clicks if direction == "up" else -clicks
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Scroll", x=ax, y=ay, amount=amount)
        pyautogui.scroll(amount, x=ax, y=ay)

    # ------------------------------------------------------------------
    # Перетаскивание
    # ------------------------------------------------------------------

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
        button: str = "left",
    ) -> None:
        """Перетаскивает из *(start_x, start_y)* в *(end_x, end_y)*.

        Args:
            start_x: Горизонтальная координата начала перетаскивания.
            start_y: Вертикальная координата начала перетаскивания.
            end_x: Горизонтальная координата конца перетаскивания.
            end_y: Вертикальная координата конца перетаскивания.
            duration: Общая длительность анимации перетаскивания в секундах.
            button: Кнопка мыши, удерживаемая при перетаскивании.
        """
        asx, asy = self._jitter(start_x, start_y)
        aex, aey = self._jitter(end_x, end_y)
        self._pre_delay()
        logger.debug("Drag", start=(asx, asy), end=(aex, aey), duration=duration)
        pyautogui.moveTo(asx, asy, duration=self.move_duration)
        pyautogui.dragTo(aex, aey, duration=duration, button=button)

    def drag_to(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
    ) -> None:
        """Перетаскивает из *(start_x, start_y)* в *(end_x, end_y)* с умолчательной скоростью.

        Удобный псевдоним для :meth:`drag` с длительностью по умолчанию.

        Args:
            start_x: Горизонтальная координата начала перетаскивания.
            start_y: Вертикальная координата начала перетаскивания.
            end_x: Горизонтальная координата конца перетаскивания.
            end_y: Вертикальная координата конца перетаскивания.
        """
        self.drag(start_x, start_y, end_x, end_y)

    # ------------------------------------------------------------------
    # Запрос позиции
    # ------------------------------------------------------------------

    def position(self) -> tuple[int, int]:
        """Возвращает текущее положение курсора мыши.

        Returns:
            Кортеж *(x, y)* с координатами экрана.
        """
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _jitter(self, x: int, y: int) -> tuple[int, int]:
        """Применяет случайное смещение координат при включённом humanize."""
        if not self.humanize:
            return x, y
        dx = random.randint(-self.humanize_offset_px, self.humanize_offset_px)
        dy = random.randint(-self.humanize_offset_px, self.humanize_offset_px)
        return x + dx, y + dy

    def _pre_delay(self) -> None:
        """Делает случайную паузу перед действием при включённом humanize."""
        if not self.humanize:
            return
        delay = (
            random.randint(self.humanize_delay_min_ms, self.humanize_delay_max_ms)
            / 1000.0
        )
        time.sleep(delay)
