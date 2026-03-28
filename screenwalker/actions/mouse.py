"""Mouse action primitives.

Wraps PyAutoGUI to provide typed, cross-platform mouse operations.
All coordinate parameters are absolute screen pixel positions unless
a :class:`~screenwalker.vision.screen_state.BBox` is given, in which case
the bounding-box centre is used.
"""

from __future__ import annotations

import time

import structlog

from screenwalker.vision.screen_state import BBox

logger = structlog.get_logger(__name__)


class MouseController:
    """Stateless mouse controller backed by PyAutoGUI.

    All methods are class methods so the controller can be used without
    instantiation, but an instance is also accepted for dependency injection
    in tests.

    Attributes:
        move_duration: Default mouse move animation duration (seconds).
        failsafe: Whether PyAutoGUI's corner failsafe is active.
    """

    def __init__(
        self,
        move_duration: float = 0.2,
        failsafe: bool = True,
    ) -> None:
        """Initialize MouseController.

        Args:
            move_duration: Duration in seconds for smooth mouse movement.
                Set to 0 for instant movement.
            failsafe: Enable PyAutoGUI corner failsafe (raises
                FailSafeException when mouse is moved to a screen corner).
        """
        self.move_duration = move_duration
        self.failsafe = failsafe
        # TODO: import pyautogui; pyautogui.FAILSAFE = failsafe

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        duration: float | None = None,
    ) -> None:
        """Move to (x, y) and click.

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
            button: Mouse button — ``"left"``, ``"right"``, or ``"middle"``.
            duration: Override move duration for this call.
        """
        d = duration if duration is not None else self.move_duration
        logger.debug("Mouse click", x=x, y=y, button=button)
        # TODO: pyautogui.click(x, y, button=button, duration=d)
        raise NotImplementedError("TODO: implement click")

    def click_bbox(self, bbox: BBox, button: str = "left") -> None:
        """Click the centre of a bounding box.

        Args:
            bbox: Target bounding box.
            button: Mouse button to use.
        """
        cx, cy = bbox.center
        self.click(cx, cy, button=button)

    def double_click(self, x: int, y: int, interval: float = 0.1) -> None:
        """Double-click at (x, y).

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
            interval: Pause between the two clicks in seconds.
        """
        logger.debug("Double click", x=x, y=y)
        # TODO: pyautogui.doubleClick(x, y, interval=interval, duration=self.move_duration)
        raise NotImplementedError("TODO: implement double_click")

    def right_click(self, x: int, y: int) -> None:
        """Right-click at (x, y).

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
        """
        self.click(x, y, button="right")

    def scroll(self, x: int, y: int, clicks: int, direction: str = "down") -> None:
        """Scroll the mouse wheel at (x, y).

        Args:
            x: Horizontal screen coordinate.
            y: Vertical screen coordinate.
            clicks: Number of scroll "ticks".
            direction: ``"up"`` or ``"down"``.

        Raises:
            ValueError: If *direction* is not ``"up"`` or ``"down"``.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got: {direction!r}")
        amount = clicks if direction == "up" else -clicks
        logger.debug("Scroll", x=x, y=y, amount=amount)
        # TODO: pyautogui.scroll(amount, x=x, y=y)
        raise NotImplementedError("TODO: implement scroll")

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
        button: str = "left",
    ) -> None:
        """Drag from (start_x, start_y) to (end_x, end_y).

        Args:
            start_x: Drag start horizontal coordinate.
            start_y: Drag start vertical coordinate.
            end_x: Drag end horizontal coordinate.
            end_y: Drag end vertical coordinate.
            duration: Total drag animation duration in seconds.
            button: Mouse button to hold during drag.
        """
        logger.debug("Drag", start=(start_x, start_y), end=(end_x, end_y))
        # TODO: pyautogui.drag(end_x - start_x, end_y - start_y,
        #                      startX=start_x, startY=start_y, duration=duration, button=button)
        raise NotImplementedError("TODO: implement drag")

    def move_to(self, x: int, y: int, duration: float | None = None) -> None:
        """Move the mouse to (x, y) without clicking.

        Args:
            x: Target horizontal coordinate.
            y: Target vertical coordinate.
            duration: Override move duration for this call.
        """
        d = duration if duration is not None else self.move_duration
        # TODO: pyautogui.moveTo(x, y, duration=d)
        raise NotImplementedError("TODO: implement move_to")

    def position(self) -> tuple[int, int]:
        """Return the current mouse cursor position.

        Returns:
            Tuple of (x, y) screen coordinates.
        """
        # TODO: return pyautogui.position()
        raise NotImplementedError("TODO: implement position")
