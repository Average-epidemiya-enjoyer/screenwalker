"""Mouse action primitives.

Wraps PyAutoGUI to provide typed, cross-platform mouse operations with
optional human-like jitter (random coordinate offset + pre-action delay).

Implements :class:`~screenwalker.actions.protocols.MouseInputProtocol` so the
controller can be replaced with an RDP- or VNC-backed implementation without
changing call sites.
"""

from __future__ import annotations

import random
import time

import pyautogui
import structlog

from screenwalker.vision.screen_state import BBox

logger = structlog.get_logger(__name__)


class MouseController:
    """PyAutoGUI-backed mouse controller with optional humanisation.

    Attributes:
        move_duration: Default mouse move animation duration (seconds).
        failsafe: Whether PyAutoGUI corner failsafe is active.
        humanize: When *True*, adds ±*humanize_offset_px* random pixel jitter
            to coordinates and a random pre-action delay between
            *humanize_delay_min_ms* and *humanize_delay_max_ms* milliseconds.
        humanize_offset_px: Maximum random pixel offset (±) per axis.
        humanize_delay_min_ms: Minimum random pre-action delay in ms.
        humanize_delay_max_ms: Maximum random pre-action delay in ms.
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
        """Initialise MouseController.

        Args:
            move_duration: Duration in seconds for smooth mouse movement.
                Set to 0 for instant movement.
            failsafe: Enable PyAutoGUI corner failsafe.
            humanize: Enable random jitter on all actions.
            humanize_offset_px: Pixel jitter magnitude (±) applied to each axis.
            humanize_delay_min_ms: Lower bound for random delay.
            humanize_delay_max_ms: Upper bound for random delay.
        """
        self.move_duration = move_duration
        self.failsafe = failsafe
        self.humanize = humanize
        self.humanize_offset_px = humanize_offset_px
        self.humanize_delay_min_ms = humanize_delay_min_ms
        self.humanize_delay_max_ms = humanize_delay_max_ms
        pyautogui.FAILSAFE = failsafe

    # ------------------------------------------------------------------
    # Click actions
    # ------------------------------------------------------------------

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Move to *(x, y)* and click.

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
            button: Mouse button — ``"left"``, ``"right"``, or ``"middle"``.
            clicks: Number of clicks to perform.
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
        """Click the centre of a bounding box.

        Args:
            bbox: Target bounding box.
            button: Mouse button to use.
        """
        cx, cy = bbox.center
        self.click(cx, cy, button=button)

    def double_click(self, x: int, y: int) -> None:
        """Double-click at *(x, y)*.

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
        """
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Double click", x=ax, y=ay)
        pyautogui.doubleClick(ax, ay, duration=self.move_duration)

    def right_click(self, x: int, y: int) -> None:
        """Right-click at *(x, y)*.

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
        """
        self.click(x, y, button="right")

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_to(self, x: int, y: int, duration: float | None = None) -> None:
        """Move the mouse cursor to *(x, y)* without clicking.

        Args:
            x: Target horizontal coordinate.
            y: Target vertical coordinate.
            duration: Animation duration in seconds.  Defaults to
                :attr:`move_duration`.
        """
        d = duration if duration is not None else self.move_duration
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Move to", x=ax, y=ay, duration=d)
        pyautogui.moveTo(ax, ay, duration=d)

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll(
        self,
        x: int,
        y: int,
        clicks: int = 3,
        direction: str = "down",
    ) -> None:
        """Scroll the mouse wheel at *(x, y)*.

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
            clicks: Number of scroll ticks.
            direction: ``"up"`` or ``"down"``.

        Raises:
            ValueError: If *direction* is not ``"up"`` or ``"down"``.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got: {direction!r}")
        amount = clicks if direction == "up" else -clicks
        ax, ay = self._jitter(x, y)
        self._pre_delay()
        logger.debug("Scroll", x=ax, y=ay, amount=amount)
        pyautogui.scroll(amount, x=ax, y=ay)

    # ------------------------------------------------------------------
    # Drag
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
        """Drag from *(start_x, start_y)* to *(end_x, end_y)*.

        Args:
            start_x: Drag start horizontal coordinate.
            start_y: Drag start vertical coordinate.
            end_x: Drag end horizontal coordinate.
            end_y: Drag end vertical coordinate.
            duration: Total drag animation duration in seconds.
            button: Mouse button to hold during drag.
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
        """Drag from *(start_x, start_y)* to *(end_x, end_y)* at default speed.

        Convenience alias for :meth:`drag` with default duration.

        Args:
            start_x: Drag start horizontal coordinate.
            start_y: Drag start vertical coordinate.
            end_x: Drag end horizontal coordinate.
            end_y: Drag end vertical coordinate.
        """
        self.drag(start_x, start_y, end_x, end_y)

    # ------------------------------------------------------------------
    # Position query
    # ------------------------------------------------------------------

    def position(self) -> tuple[int, int]:
        """Return the current mouse cursor position.

        Returns:
            Tuple of *(x, y)* screen coordinates.
        """
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _jitter(self, x: int, y: int) -> tuple[int, int]:
        """Apply random coordinate offset when humanize is enabled."""
        if not self.humanize:
            return x, y
        dx = random.randint(-self.humanize_offset_px, self.humanize_offset_px)
        dy = random.randint(-self.humanize_offset_px, self.humanize_offset_px)
        return x + dx, y + dy

    def _pre_delay(self) -> None:
        """Sleep a random interval before an action when humanize is enabled."""
        if not self.humanize:
            return
        delay = (
            random.randint(self.humanize_delay_min_ms, self.humanize_delay_max_ms)
            / 1000.0
        )
        time.sleep(delay)
