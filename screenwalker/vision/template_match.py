"""Сопоставление шаблонов OpenCV — одномасштабное, многомасштабное и NMS.

Публичный API:
- :class:`TemplateMatcher` — классовый API с директорией шаблонов и кэшированием.
- :func:`match_template` / :func:`match_template_all` /
  :func:`match_template_multiscale` — автономные функции для разовых вызовов.
- :func:`_nms` — чистый numpy Non-Maximum Suppression (экспортируется для тестов).
- :func:`draw_matches` — визуализация для отладки.
- :func:`capture_template` — вспомогательный инструмент интерактивного захвата области.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import structlog
from PIL import Image, ImageDraw

from screenwalker.vision.screen_state import BBox, FindResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции преобразования (сохранены для обратной совместимости)
# ---------------------------------------------------------------------------


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Преобразовать PIL Image в массив OpenCV BGR uint8.

    Args:
        image: Исходное PIL Image (любой режим).

    Returns:
        Массив ndarray BGR uint8, совместимый с OpenCV, формы (H, W, 3).
    """
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    """Преобразовать массив BGR в оттенки серого.

    Args:
        bgr: Массив BGR uint8 формы (H, W, 3).

    Returns:
        Массив в оттенках серого uint8 формы (H, W).
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _load_template(template_path: Path | str) -> np.ndarray:
    """Загрузить изображение шаблона с диска как массив BGR uint8.

    Args:
        template_path: Путь к файлу шаблона PNG/JPEG.

    Returns:
        Изображение шаблона как массив ndarray BGR uint8.

    Raises:
        FileNotFoundError: Если файл не существует.
        ValueError: Если файл не удаётся декодировать как изображение.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    template = cv2.imread(str(path))
    if template is None:
        raise ValueError(f"Could not decode template image: {path}")
    return template


# ---------------------------------------------------------------------------
# Датакласс MatchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """Результат одного совпадения шаблона.

    Attributes:
        template_name: Имя (stem) совпавшего шаблона.
        confidence: Нормализованный показатель совпадения в диапазоне [0.0, 1.0].
        center: Пиксельные координаты центра совпадения ``(x, y)``.
        bbox: Ограничивающий прямоугольник совпадения в координатах изображения.
        scale: Масштаб шаблона, при котором получено данное совпадение (1.0 = исходный размер).
    """

    template_name: str
    confidence: float
    center: tuple[int, int]
    bbox: BBox
    scale: float = 1.0


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]:
    """Чистый numpy Non-Maximum Suppression.

    Args:
        boxes: Массив формы ``(N, 4)`` со столбцами ``[x, y, w, h]``.
        scores: Массив формы ``(N,)`` с показателями уверенности.
        iou_threshold: Прямоугольники, у которых IoU с текущим выбранным
            прямоугольником превышает это значение, подавляются.

    Returns:
        Список сохранённых индексов в *boxes*, отсортированных по убыванию оценки.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = (boxes[:, 0] + boxes[:, 2]).astype(float)
    y2 = (boxes[:, 1] + boxes[:, 3]).astype(float)
    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-9)

        order = rest[iou <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# Класс TemplateMatcher
# ---------------------------------------------------------------------------


class TemplateMatcher:
    """Классовый сопоставитель шаблонов с управлением директорией шаблонов и кэшированием.

    Загруженные шаблоны кэшируются в памяти, поэтому повторные вызовы
    :meth:`find_one` / :meth:`find_all` для одного и того же имени шаблона не
    выполняют повторное чтение с диска.

    Attributes:
        templates_dir: Корневая директория, содержащая эталонные PNG-шаблоны.
        method: Метод сопоставления OpenCV (по умолчанию ``cv2.TM_CCOEFF_NORMED``).
    """

    def __init__(
        self,
        templates_dir: Path,
        method: int = cv2.TM_CCOEFF_NORMED,
    ) -> None:
        """Инициализировать TemplateMatcher.

        Args:
            templates_dir: Директория, содержащая файлы шаблонов ``.png``.
            method: Константа метода ``matchTemplate`` OpenCV.
        """
        self.templates_dir = Path(templates_dir)
        self.method = method
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Загрузка шаблонов
    # ------------------------------------------------------------------

    def load_template(self, name: str) -> np.ndarray:
        """Загрузить шаблон по имени с кэшированием в памяти.

        Сначала пробует ``templates_dir/<name>``, затем ``templates_dir/<name>.png``.

        Args:
            name: Stem шаблона (например ``"ok_button"``) или полное имя файла.

        Returns:
            Массив ndarray BGR uint8.

        Raises:
            FileNotFoundError: Если ни один из кандидатов пути не существует.
        """
        if name in self._cache:
            return self._cache[name]

        for candidate in [name, f"{name}.png"]:
            path = self.templates_dir / candidate
            if path.exists():
                self._cache[name] = _load_template(path)
                logger.debug("Loaded template", name=name, path=str(path))
                return self._cache[name]

        raise FileNotFoundError(
            f"Template '{name}' not found in {self.templates_dir}. "
            f"Tried: {name}, {name}.png"
        )

    def clear_cache(self) -> None:
        """Удалить все кэшированные шаблоны из памяти."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Одномасштабное сопоставление
    # ------------------------------------------------------------------

    def find_one(
        self,
        screenshot: Image.Image,
        template_name: str,
        threshold: float = 0.8,
        region: BBox | None = None,
    ) -> MatchResult | None:
        """Найти единственное наилучшее совпадение шаблона на скриншоте.

        Args:
            screenshot: PIL Image всего экрана или окна.
            template_name: Stem шаблона для загрузки из :attr:`templates_dir`.
            threshold: Минимальный нормализованный показатель совпадения.
            region: Опциональный ограничивающий прямоугольник для ограничения области поиска.

        Returns:
            :class:`MatchResult` для наилучшего совпадения, или None.
        """
        tmpl = self.load_template(template_name)
        gray_screen, ox, oy = self._prepare_screen(screenshot, region)
        gray_tmpl = _to_gray(tmpl)

        hit = self._match_best(gray_screen, gray_tmpl, threshold)
        if hit is None:
            return None

        conf, x, y = hit
        th, tw = gray_tmpl.shape[:2]
        logger.debug("Template found", name=template_name, confidence=conf, x=ox + x, y=oy + y)
        return MatchResult(
            template_name=template_name,
            confidence=conf,
            center=(ox + x + tw // 2, oy + y + th // 2),
            bbox=BBox(x=ox + x, y=oy + y, w=tw, h=th),
            scale=1.0,
        )

    def find_all(
        self,
        screenshot: Image.Image,
        template_name: str,
        threshold: float = 0.8,
        region: BBox | None = None,
    ) -> list[MatchResult]:
        """Найти все непересекающиеся совпадения шаблона на скриншоте.

        Применяет Non-Maximum Suppression (IoU > 0.5) для удаления дублирующих
        обнаружений из перекрывающихся пиков карты совпадений.

        Args:
            screenshot: PIL Image всего экрана или окна.
            template_name: Stem шаблона для загрузки.
            threshold: Минимальный показатель совпадения.
            region: Опциональная область поиска.

        Returns:
            Список :class:`MatchResult`, отсортированный по убыванию уверенности.
        """
        tmpl = self.load_template(template_name)
        gray_screen, ox, oy = self._prepare_screen(screenshot, region)
        gray_tmpl = _to_gray(tmpl)

        th, tw = gray_tmpl.shape[:2]
        raw = self._match_all(gray_screen, gray_tmpl, threshold)
        if not raw:
            return []

        boxes = np.array([[x, y, tw, th] for _, x, y in raw], dtype=float)
        scores = np.array([c for c, _, _ in raw])
        keep = _nms(boxes, scores, iou_threshold=0.5)

        matches = [
            MatchResult(
                template_name=template_name,
                confidence=raw[i][0],
                center=(ox + raw[i][1] + tw // 2, oy + raw[i][2] + th // 2),
                bbox=BBox(x=ox + raw[i][1], y=oy + raw[i][2], w=tw, h=th),
                scale=1.0,
            )
            for i in keep
        ]
        logger.debug(
            "Template find_all", name=template_name, count=len(matches)
        )
        return sorted(matches, key=lambda r: r.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Многомасштабное сопоставление
    # ------------------------------------------------------------------

    def find_one_multiscale(
        self,
        screenshot: Image.Image,
        template_name: str,
        scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
        threshold: float = 0.75,
        region: BBox | None = None,
    ) -> MatchResult | None:
        """Найти наилучшее совпадение при нескольких масштабах шаблона.

        Изменяет размер *шаблона* (не скриншота) на каждом уровне масштаба и
        выбирает масштаб, дающий наибольшую уверенность выше *threshold*.
        Полезно для обработки различий DPI / разрешения между эталонным
        скриншотом и живым экраном.

        Args:
            screenshot: PIL Image всего экрана или окна.
            template_name: Stem шаблона для загрузки.
            scales: Кортеж масштабных коэффициентов для проверки (относительно исходного размера).
            threshold: Минимальный показатель совпадения для признания успешным.
            region: Опциональная область поиска.

        Returns:
            :class:`MatchResult` для наилучшего совпадения по всем масштабам, или None.
        """
        tmpl = self.load_template(template_name)
        gray_screen, ox, oy = self._prepare_screen(screenshot, region)
        gray_tmpl_orig = _to_gray(tmpl)
        h_orig, w_orig = gray_tmpl_orig.shape[:2]

        best: MatchResult | None = None

        for scale in scales:
            new_w = max(1, int(w_orig * scale))
            new_h = max(1, int(h_orig * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            gray_tmpl = cv2.resize(
                gray_tmpl_orig, (new_w, new_h), interpolation=interp
            )

            hit = self._match_best(gray_screen, gray_tmpl, threshold)
            if hit is None:
                continue

            conf, x, y = hit
            candidate = MatchResult(
                template_name=template_name,
                confidence=conf,
                center=(ox + x + new_w // 2, oy + y + new_h // 2),
                bbox=BBox(x=ox + x, y=oy + y, w=new_w, h=new_h),
                scale=scale,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        if best is not None:
            logger.debug(
                "Multiscale match",
                name=template_name,
                scale=best.scale,
                confidence=best.confidence,
            )
        return best

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _prepare_screen(
        self,
        screenshot: Image.Image,
        region: BBox | None,
    ) -> tuple[np.ndarray, int, int]:
        """Вернуть ``(gray_array, offset_x, offset_y)``."""
        if region is not None:
            src = screenshot.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            ox, oy = region.x, region.y
        else:
            src = screenshot
            ox, oy = 0, 0
        return _to_gray(_pil_to_cv(src)), ox, oy

    def _match_best(
        self,
        gray_screen: np.ndarray,
        gray_tmpl: np.ndarray,
        threshold: float,
    ) -> tuple[float, int, int] | None:
        """Вернуть ``(conf, x, y)`` для наилучшего совпадения, или None."""
        if (
            gray_tmpl.shape[0] > gray_screen.shape[0]
            or gray_tmpl.shape[1] > gray_screen.shape[1]
        ):
            return None
        result = cv2.matchTemplate(gray_screen, gray_tmpl, self.method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return float(max_val), int(max_loc[0]), int(max_loc[1])
        return None

    def _match_all(
        self,
        gray_screen: np.ndarray,
        gray_tmpl: np.ndarray,
        threshold: float,
    ) -> list[tuple[float, int, int]]:
        """Вернуть все совпадения ``(conf, x, y)`` выше *threshold*."""
        if (
            gray_tmpl.shape[0] > gray_screen.shape[0]
            or gray_tmpl.shape[1] > gray_screen.shape[1]
        ):
            return []
        result = cv2.matchTemplate(gray_screen, gray_tmpl, self.method)
        ys, xs = np.where(result >= threshold)
        return [(float(result[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def draw_matches(
    screenshot: Image.Image,
    matches: list[MatchResult],
    box_color: tuple[int, int, int] = (0, 220, 0),
    label_color: tuple[int, int, int] = (255, 50, 50),
    line_width: int = 2,
) -> Image.Image:
    """Нарисовать ограничивающие прямоугольники и метки на копии *screenshot*.

    Args:
        screenshot: Исходное изображение (не изменяется на месте).
        matches: Список :class:`MatchResult` для отрисовки.
        box_color: Цвет RGB для контура ограничивающего прямоугольника.
        label_color: Цвет RGB для метки уверенности.
        line_width: Толщина контура в пикселях.

    Returns:
        Новое PIL Image с аннотациями.
    """
    img = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for m in matches:
        b = m.bbox
        draw.rectangle(
            [b.x, b.y, b.x + b.w, b.y + b.h],
            outline=box_color,
            width=line_width,
        )
        label = f"{m.template_name} {m.confidence:.2f}"
        draw.text((b.x + 2, max(0, b.y - 12)), label, fill=label_color)
    return img


def capture_template(
    name: str,
    region: BBox,
    templates_dir: Path | str = Path("templates"),
) -> Path:
    """Захватить область экрана и сохранить её как PNG-шаблон.

    Делает скриншот, обрезает его до *region* и сохраняет результат как
    ``<templates_dir>/<name>.png``.

    Args:
        name: Stem шаблона (без расширения).
        region: Область экрана для захвата в абсолютных координатах экрана.
        templates_dir: Целевая директория (создаётся, если не существует).

    Returns:
        Абсолютный путь сохранённого файла шаблона.
    """
    import pyautogui  # отложенный импорт, чтобы модуль был импортируемым без дисплея

    dest = Path(templates_dir)
    dest.mkdir(parents=True, exist_ok=True)

    screenshot = pyautogui.screenshot()
    cropped = screenshot.crop(
        (region.x, region.y, region.right, region.bottom)
    )
    path = (dest / f"{name}.png").resolve()
    cropped.save(path, "PNG")
    logger.info("Saved template", name=name, path=str(path))
    return path


# ---------------------------------------------------------------------------
# Автономные функции (обратно совместимый API на основе путей)
# ---------------------------------------------------------------------------


def match_template(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    region: BBox | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> FindResult | None:
    """Одномасштабное сопоставление шаблонов с использованием OpenCV.

    Args:
        image: Скриншот для поиска.
        template_path: Путь к эталонному изображению шаблона.
        threshold: Минимальный нормализованный показатель совпадения (0.0–1.0).
        region: Опциональный ограничивающий прямоугольник для обрезки перед сопоставлением.
        method: Константа метода сопоставления OpenCV.

    Returns:
        :class:`FindResult` для наилучшего совпадения выше порога, или None.
    """
    path = Path(template_path)
    tmpl = _load_template(path)

    if region is not None:
        src = image.crop((region.x, region.y, region.right, region.bottom))
        ox, oy = region.x, region.y
    else:
        src = image
        ox, oy = 0, 0

    gray_screen = _to_gray(_pil_to_cv(src))
    gray_tmpl = _to_gray(tmpl)
    th, tw = gray_tmpl.shape[:2]

    if th > gray_screen.shape[0] or tw > gray_screen.shape[1]:
        return None

    result = cv2.matchTemplate(gray_screen, gray_tmpl, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    x, y = max_loc
    return FindResult(
        element=path.stem,
        confidence=float(max_val),
        bbox=BBox(x=ox + x, y=oy + y, w=tw, h=th),
        method="template",
        metadata={"scale": 1.0},
    )


def match_template_all(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    region: BBox | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
    max_results: int = 50,
) -> list[FindResult]:
    """Найти ВСЕ непересекающиеся вхождения шаблона в изображении.

    Применяет Non-Maximum Suppression (IoU > 0.5) для устранения дублирующих
    обнаружений.

    Args:
        image: Скриншот для поиска.
        template_path: Путь к эталонному изображению шаблона.
        threshold: Минимальный показатель совпадения.
        region: Опциональная область поиска.
        method: Константа метода сопоставления OpenCV.
        max_results: Максимальное количество возвращаемых результатов.

    Returns:
        Список :class:`FindResult`, отсортированный по убыванию уверенности.
    """
    path = Path(template_path)
    tmpl = _load_template(path)

    if region is not None:
        src = image.crop((region.x, region.y, region.right, region.bottom))
        ox, oy = region.x, region.y
    else:
        src = image
        ox, oy = 0, 0

    gray_screen = _to_gray(_pil_to_cv(src))
    gray_tmpl = _to_gray(tmpl)
    th, tw = gray_tmpl.shape[:2]

    if th > gray_screen.shape[0] or tw > gray_screen.shape[1]:
        return []

    result_map = cv2.matchTemplate(gray_screen, gray_tmpl, method)
    ys, xs = np.where(result_map >= threshold)
    raw = [(float(result_map[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
    if not raw:
        return []

    boxes = np.array([[x, y, tw, th] for _, x, y in raw], dtype=float)
    scores = np.array([c for c, _, _ in raw])
    keep = _nms(boxes, scores, iou_threshold=0.5)

    out: list[FindResult] = []
    for i in keep[:max_results]:
        conf, x, y = raw[i]
        out.append(
            FindResult(
                element=path.stem,
                confidence=conf,
                bbox=BBox(x=ox + x, y=oy + y, w=tw, h=th),
                method="template",
                metadata={"scale": 1.0},
            )
        )
    return sorted(out, key=lambda r: r.confidence, reverse=True)


def match_template_multiscale(
    image: Image.Image,
    template_path: Path | str,
    threshold: float = 0.80,
    scale_min: float = 0.7,
    scale_max: float = 1.3,
    scale_steps: int = 10,
    region: BBox | None = None,
) -> FindResult | None:
    """Многомасштабное сопоставление шаблонов.

    Выполняет поиск при нескольких размерах шаблона для обработки различий
    масштаба/DPI между эталонным скриншотом и живым экраном.

    Args:
        image: Скриншот для поиска.
        template_path: Путь к эталонному изображению шаблона.
        threshold: Минимальный показатель совпадения.
        scale_min: Наименьший масштабный коэффициент относительно исходного шаблона.
        scale_max: Наибольший масштабный коэффициент относительно исходного шаблона.
        scale_steps: Количество равномерно распределённых уровней масштаба для проверки.
        region: Опциональная область поиска.

    Returns:
        :class:`FindResult` для наилучшего совпадения по всем масштабам, или None.
    """
    path = Path(template_path)
    tmpl = _load_template(path)

    if region is not None:
        src = image.crop((region.x, region.y, region.right, region.bottom))
        ox, oy = region.x, region.y
    else:
        src = image
        ox, oy = 0, 0

    gray_screen = _to_gray(_pil_to_cv(src))
    gray_tmpl_orig = _to_gray(tmpl)
    h_orig, w_orig = gray_tmpl_orig.shape[:2]

    best_conf = -1.0
    best: FindResult | None = None

    for scale in np.linspace(scale_min, scale_max, scale_steps):
        scale = float(scale)
        new_w = max(1, int(w_orig * scale))
        new_h = max(1, int(h_orig * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray_tmpl = cv2.resize(
            gray_tmpl_orig, (new_w, new_h), interpolation=interp
        )

        if new_h > gray_screen.shape[0] or new_w > gray_screen.shape[1]:
            continue

        result_map = cv2.matchTemplate(
            gray_screen, gray_tmpl, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result_map)

        if max_val >= threshold and max_val > best_conf:
            best_conf = max_val
            x, y = max_loc
            best = FindResult(
                element=path.stem,
                confidence=float(max_val),
                bbox=BBox(x=ox + x, y=oy + y, w=new_w, h=new_h),
                method="template",
                metadata={"scale": scale},
            )

    return best
