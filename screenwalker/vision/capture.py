"""Утилиты захвата экрана и окон.

Предоставляет как класс :class:`ScreenCapture` (рекомендуется), так и
модульные функции-обёртки для делегирования к экземпляру по умолчанию.

Все возвращаемые изображения — объекты ``PIL.Image.Image`` в режиме RGB.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import pyautogui
import structlog
from PIL import Image

from screenwalker.vision.screen_state import BBox

logger = structlog.get_logger(__name__)


class WindowInfo(NamedTuple):
    """Базовые метаданные о захваченном окне.

    Attributes:
        title: Строка заголовка окна.
        bbox: Положение и размер окна на экране.
        pid: Идентификатор процесса ОС, владеющего окном (если доступен).
    """

    title: str
    bbox: BBox
    pid: int | None


class ScreenCapture:
    """Кросс-платформенный помощник для скриншотов с кэшированием изображений по шагам.

    Attributes:
        screenshot_delay: Секунды паузы перед каждым захватом (время стабилизации UI).
        debug_dir: Директория для вывода :meth:`save_debug`.
    """

    def __init__(
        self,
        screenshot_delay: float = 0.3,
        debug_dir: str | Path = "logs",
    ) -> None:
        """Инициализация ScreenCapture.

        Args:
            screenshot_delay: Пауза перед захватом, чтобы UI успел стабилизироваться.
            debug_dir: Корневая директория для отладочных скриншотов,
                записываемых :meth:`save_debug`.
        """
        self.screenshot_delay = screenshot_delay
        self.debug_dir = Path(debug_dir)
        self._last_image: Image.Image | None = None

    # ------------------------------------------------------------------
    # Публичный API захвата
    # ------------------------------------------------------------------

    def capture_full(self) -> Image.Image:
        """Захватывает весь основной экран.

        Returns:
            Скриншот полного экрана в виде PIL-образа в режиме RGB.
        """
        self._settle()
        img = pyautogui.screenshot()
        img = img.convert("RGB")
        self._last_image = img
        logger.debug("Captured full screen", size=img.size)
        return img

    def capture_region(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """Захватывает прямоугольную область экрана.

        Args:
            x: Левый край в пикселях экрана.
            y: Верхний край в пикселях экрана.
            w: Ширина в пикселях.
            h: Высота в пикселях.

        Returns:
            Обрезанный скриншот в виде PIL-образа в режиме RGB.
        """
        self._settle()
        img = pyautogui.screenshot(region=(x, y, w, h))
        img = img.convert("RGB")
        self._last_image = img
        logger.debug("Captured region", x=x, y=y, w=w, h=h)
        return img

    def capture_window(self, title: str) -> Image.Image:
        """Захватывает окно, заголовок которого содержит *title*.

        Выполняет регистронезависимый поиск по части заголовка и
        активирует окно перед захватом.

        Args:
            title: Часть или полный заголовок окна (без учёта регистра).

        Returns:
            Скриншот окна в виде PIL-образа в режиме RGB.

        Raises:
            ValueError: Если окно с заголовком *title* не найдено.
            RuntimeError: Если необходимая платформенная библиотека недоступна.
        """
        system = platform.system()
        if system == "Windows":
            return self._capture_window_windows(title)
        elif system == "Darwin":
            return self._capture_window_macos(title)
        else:
            return self._capture_window_linux(title)

    # ------------------------------------------------------------------
    # Кэш последнего изображения
    # ------------------------------------------------------------------

    @property
    def last_image(self) -> Image.Image | None:
        """Последний захваченный образ или None, если захвата ещё не было."""
        return self._last_image

    # ------------------------------------------------------------------
    # Отладочные вспомогательные методы
    # ------------------------------------------------------------------

    def save_debug(self, image: Image.Image, step_name: str) -> Path:
        """Сохраняет *image* в директорию отладки с очищенным именем файла.

        Args:
            image: PIL-образ для сохранения.
            step_name: Читаемая метка, используемая как имя файла.

        Returns:
            Абсолютный путь к записанному файлу.
        """
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        safe_name = step_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        path = self.debug_dir / f"{safe_name}.png"
        image.save(path, "PNG")
        logger.debug("Saved debug screenshot", path=str(path), step=step_name)
        return path.resolve()

    # ------------------------------------------------------------------
    # Платформозависимый захват окна
    # ------------------------------------------------------------------

    def _capture_window_windows(self, title: str) -> Image.Image:
        try:
            import pygetwindow as gw  # устанавливается автоматически с pyautogui на Windows
        except ImportError as exc:
            raise RuntimeError(
                "pygetwindow is required for window capture on Windows. "
                "It is normally installed automatically with pyautogui."
            ) from exc

        windows = gw.getWindowsWithTitle(title)
        if not windows:
            raise ValueError(f"No window found with title matching: {title!r}")

        win = windows[0]
        try:
            win.activate()
            time.sleep(0.2)
        except Exception:
            pass  # неудача активации некритична; захватываем то, что видно

        x, y, w, h = win.left, win.top, win.width, win.height
        self._settle()
        img = pyautogui.screenshot(region=(x, y, w, h))
        img = img.convert("RGB")
        self._last_image = img
        logger.debug("Captured window (Windows)", title=title, bbox=(x, y, w, h))
        return img

    def _capture_window_macos(self, title: str) -> Image.Image:  # pragma: no cover
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PyObjC Quartz bindings are required for window capture on macOS: "
                "pip install pyobjc-framework-Quartz"
            ) from exc

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        title_lower = title.lower()
        match = next(
            (
                w
                for w in window_list
                if title_lower in (w.get("kCGWindowName") or "").lower()
            ),
            None,
        )
        if match is None:
            raise ValueError(f"No window found with title matching: {title!r}")

        bounds = match["kCGWindowBounds"]
        x = int(bounds["X"])
        y = int(bounds["Y"])
        w = int(bounds["Width"])
        h = int(bounds["Height"])
        return self.capture_region(x, y, w, h)

    def _capture_window_linux(self, title: str) -> Image.Image:  # pragma: no cover
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "xdotool is required for window capture on Linux: "
                "sudo apt install xdotool"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"No window found with title matching: {title!r}"
            ) from exc

        wid = result.stdout.strip().split("\n")[0]
        geom_result = subprocess.run(
            ["xdotool", "getwindowgeometry", "--shell", wid],
            capture_output=True,
            text=True,
            check=True,
        )
        geom: dict[str, int] = {}
        for line in geom_result.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                geom[k.strip()] = int(v.strip())

        return self.capture_region(geom["X"], geom["Y"], geom["WIDTH"], geom["HEIGHT"])

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _settle(self) -> None:
        if self.screenshot_delay > 0:
            time.sleep(self.screenshot_delay)


# ---------------------------------------------------------------------------
# Модульные функции-обёртки (обратная совместимость)
# ---------------------------------------------------------------------------

_default_capture = ScreenCapture(screenshot_delay=0.0)


def capture_screen(delay: float = 0.0) -> Image.Image:
    """Захватывает весь основной экран.

    Args:
        delay: Опциональная пауза в секундах перед захватом.

    Returns:
        Скриншот полного экрана в виде PIL-образа в режиме RGB.
    """
    _default_capture.screenshot_delay = delay
    return _default_capture.capture_full()


def capture_region(bbox: BBox, delay: float = 0.0) -> Image.Image:
    """Захватывает прямоугольную область экрана.

    Args:
        bbox: Область экрана для захвата (x, y, w, h в пикселях экрана).
        delay: Опциональная пауза перед захватом.

    Returns:
        Обрезанный скриншот в виде PIL-образа в режиме RGB.
    """
    _default_capture.screenshot_delay = delay
    return _default_capture.capture_region(bbox.x, bbox.y, bbox.w, bbox.h)


def capture_window(title: str, delay: float = 0.0) -> Image.Image:
    """Захватывает окно с заданным заголовком.

    Args:
        title: Часть или полный заголовок окна (без учёта регистра).
        delay: Опциональная пауза перед захватом.

    Returns:
        Скриншот окна в виде PIL-образа в режиме RGB.

    Raises:
        ValueError: Если окно с заголовком *title* не найдено.
    """
    _default_capture.screenshot_delay = delay
    return _default_capture.capture_window(title)


def list_windows() -> list[WindowInfo]:
    """Возвращает метаданные всех видимых окон верхнего уровня.

    Returns:
        Список :class:`WindowInfo`. На неподдерживаемых платформах возвращает
        пустой список вместо исключения.
    """
    system = platform.system()
    if system == "Windows":
        try:
            import pygetwindow as gw

            result: list[WindowInfo] = []
            for win in gw.getAllWindows():
                if win.title:
                    bbox = BBox(x=win.left, y=win.top, w=win.width, h=win.height)
                    result.append(WindowInfo(title=win.title, bbox=bbox, pid=None))
            return result
        except ImportError:
            return []
    return []


def save_image(image: Image.Image, path: Path | str, quality: int = 95) -> Path:
    """Сохраняет PIL-образ на диск в формате PNG или JPEG.

    Args:
        image: Образ для сохранения.
        path: Путь к файлу назначения. Расширение определяет формат.
        quality: Качество JPEG (1–95); игнорируется для PNG.

    Returns:
        Абсолютный разрешённый путь к сохранённому файлу.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(path, "JPEG", quality=quality)
    else:
        image.save(path, "PNG")
    return path
