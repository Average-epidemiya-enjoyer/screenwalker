"""Детектор UI-элементов на основе YOLO.

Доступны три варианта, выбираемые автоматически функцией :func:`create_detector`:

* **UIElementDetector** — обёртка над любой YOLO-совместимой моделью (OmniParser V2,
  дообученный YOLOv8n или любые веса, совместимые с ``ultralytics``).  Требует
  дополнительного пакета ``yolo``: ``pip install screenwalker[yolo]``.

* **NullDetector** — возвращает пустые результаты без каких-либо зависимостей.
  Используется, если модель не настроена или ``ultralytics`` не установлен.

Рекомендуемая модель: **OmniParser V2** с Hugging Face
(``microsoft/OmniParser-v2.0``, ``icon_detect/best.pt``).
Загрузка: ``python scripts/download_model.py``.

Дообучение на своих данных: ``python scripts/train_detector.py``.

Нормализация классов
--------------------
Сырые имена классов YOLO приводятся к каноническому набору типов UI-элементов:

    ``button``, ``text_field``, ``icon``, ``checkbox``, ``radio_button``,
    ``dropdown``, ``link``, ``image``, ``label``, ``toggle``, ``slider``

Неизвестные имена классов сохраняются как есть.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from screenwalker.vision.screen_state import BBox, FindResult

if TYPE_CHECKING:
    from screenwalker.utils.config import VisionConfig


# ---------------------------------------------------------------------------
# Таксономия классов — сырые имена модели → канонический тип UI-элемента
# ---------------------------------------------------------------------------

UI_CLASS_ALIASES: dict[str, list[str]] = {
    "button": ["button", "btn", "push_button", "icon_button", "clickable"],
    "text_field": ["text_field", "input", "textbox", "edit", "text_input", "entry", "edittext"],
    "icon": ["icon", "symbol", "glyph", "pictogram"],
    "checkbox": ["checkbox", "check_box", "check", "tick_box"],
    "radio_button": ["radio", "radio_button", "radiobutton", "option_button"],
    "dropdown": ["dropdown", "select", "combobox", "combo", "list_box", "spinner"],
    "link": ["link", "hyperlink", "url", "anchor", "a"],
    "image": ["image", "img", "picture", "photo", "bitmap"],
    "label": ["label", "text", "static_text", "static"],
    "toggle": ["toggle", "switch", "on_off"],
    "slider": ["slider", "range", "seekbar", "progress"],
}

# Обратная таблица поиска: псевдоним/сырое имя → канонический тип
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in UI_CLASS_ALIASES.items()
    for alias in aliases
}
# Канонические имена также отображаются сами на себя
for _c in list(UI_CLASS_ALIASES):
    _ALIAS_TO_CANONICAL[_c] = _c


def normalise_class(raw: str) -> str:
    """Привести сырое имя класса модели к каноническому типу UI-элемента.

    Args:
        raw: Имя класса, возвращённое моделью (например, ``"btn"``, ``"input"``).

    Returns:
        Строка канонического типа или *raw*, если соответствие не найдено.
    """
    return _ALIAS_TO_CANONICAL.get(raw.lower().strip(), raw.lower().strip())


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionResult:
    """Единичный результат обнаружения UI-элемента.

    Attributes:
        class_name: Канонический тип UI-элемента (``"button"``, ``"text_field"``, …).
        confidence: Уверенность модели в диапазоне ``[0.0, 1.0]``.
        bbox: Ограничивающий прямоугольник в экранных координатах.
    """

    class_name: str
    confidence: float
    bbox: BBox

    @property
    def center(self) -> tuple[int, int]:
        """Пиксельные координаты центра обнаруженного элемента."""
        return self.bbox.center

    def to_find_result(self) -> FindResult:
        """Преобразовать в :class:`~screenwalker.vision.screen_state.FindResult`.

        Returns:
            FindResult с методом ``"yolo"`` и элементом, равным class_name.
        """
        return FindResult(
            element=self.class_name,
            confidence=self.confidence,
            bbox=self.bbox,
            method="yolo",
            metadata={"yolo_class": self.class_name},
        )


# ---------------------------------------------------------------------------
# UIElementDetector
# ---------------------------------------------------------------------------


class UIElementDetector:
    """Обнаружение UI-элементов с помощью YOLO-совместимой модели.

    Поддерживает любой файл весов, совместимый с ``ultralytics``, включая
    OmniParser V2 и кастомные дообученные модели YOLOv8.

    Attributes:
        model_path: Путь к файлу весов ``*.pt`` или ``None`` для режима заглушки.
        confidence_threshold: Минимальная уверенность обнаружения (0.0–1.0).
        class_names: Словарь ``{индекс_класса → сырое_имя_класса}``,
            считанный из загруженной модели.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Инициализировать детектор.

        Не падает, если ``ultralytics`` не установлен — свойство ``available``
        вернёт ``False``, а все вызовы find будут возвращать пустые результаты.

        Args:
            model_path: Путь к весам YOLO (``*.pt``).  ``None`` оставляет
                детектор в режиме заглушки.
            confidence_threshold: Минимальная уверенность для принятия обнаружения.
        """
        self.model_path: Path | None = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self._model: Any = None
        self.class_names: dict[int, str] = {}

        if self.model_path is not None:
            self._load_model()

    @property
    def available(self) -> bool:
        """``True`` если модель загружена и инференс возможен."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Загрузка модели
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Загрузить YOLO-модель из :attr:`model_path`.

        Raises:
            ImportError: Если ``ultralytics`` не установлен.
            FileNotFoundError: Если файл весов не существует.
        """
        if self.model_path is None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.model_path}\n"
                "Run: python scripts/download_model.py"
            )
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO detection: "
                "pip install screenwalker[yolo]"
            ) from exc

        self._model = YOLO(str(self.model_path))
        self.class_names = dict(self._model.names)

    # ------------------------------------------------------------------
    # Основной инференс
    # ------------------------------------------------------------------

    def detect(
        self,
        screenshot: Image.Image,
        region: BBox | None = None,
    ) -> list[DetectionResult]:
        """Обнаружить все UI-элементы на *screenshot*.

        Args:
            screenshot: Полноэкранное или обрезанное PIL-изображение.
            region: Необязательный BBox, ограничивающий область инференса.
                Возвращаемые BBox переводятся обратно в координаты полного изображения.

        Returns:
            Список :class:`DetectionResult`, отсортированный по убыванию уверенности.
            Пустой, если модель недоступна.
        """
        if not self.available:
            return []

        search_image = screenshot
        offset_x, offset_y = 0, 0
        if region is not None:
            search_image = screenshot.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            offset_x, offset_y = region.x, region.y

        img_array = np.array(search_image.convert("RGB"))

        try:
            results = self._model.predict(
                img_array,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception:
            return []

        detections: list[DetectionResult] = []
        for result in results:
            for box in result.boxes:
                try:
                    cls_idx = int(box.cls.item())
                    raw_name = self.class_names.get(cls_idx, str(cls_idx))
                    class_name = normalise_class(raw_name)
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    bbox = BBox(
                        x=int(x1) + offset_x,
                        y=int(y1) + offset_y,
                        w=max(1, int(x2 - x1)),
                        h=max(1, int(y2 - y1)),
                    )
                    detections.append(DetectionResult(
                        class_name=class_name,
                        confidence=conf,
                        bbox=bbox,
                    ))
                except Exception:
                    continue

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Поиск с фильтрацией
    # ------------------------------------------------------------------

    def find_by_class(
        self,
        screenshot: Image.Image,
        element_class: str,
        region: BBox | None = None,
    ) -> list[DetectionResult]:
        """Обнаружить все элементы заданного UI-класса.

        Проверяются как сырое имя класса модели, так и канонический псевдоним,
        поэтому ``"btn"`` и ``"button"`` одинаково соответствуют кнопке.

        Args:
            screenshot: Скриншот для анализа.
            element_class: Тип UI-элемента для фильтрации (например, ``"button"``).
            region: Необязательная область поиска.

        Returns:
            Совпадающие обнаружения, отсортированные по убыванию уверенности.
        """
        canonical_target = normalise_class(element_class)
        all_detections = self.detect(screenshot, region=region)
        return [
            d for d in all_detections
            if d.class_name == canonical_target
            or normalise_class(d.class_name) == canonical_target
        ]

    def find_nearest(
        self,
        screenshot: Image.Image,
        element_class: str,
        anchor_text: str,
        ocr_engine: Any,
    ) -> DetectionResult | None:
        """Найти UI-элемент типа *element_class*, ближайший к *anchor_text*.

        Сценарий использования: найти ``"button"``, ближайшую к тексту
        ``"Confirm deletion?"`` чтобы нажать правильную кнопку OK в диалоге.

        Шаги:

        1. Запустить OCR для нахождения *anchor_text* и получения координат его центра.
        2. Обнаружить все элементы типа *element_class* через YOLO.
        3. Вернуть обнаружение с наименьшим евклидовым расстоянием от центра
           опорного текста.

        Args:
            screenshot: Скриншот для поиска.
            element_class: Тип UI-элемента (например, ``"button"``).
            anchor_text: Текст, используемый как пространственная точка привязки.
            ocr_engine: OCR-движок, реализующий ``find_text(image, query)``.

        Returns:
            :class:`DetectionResult` для ближайшего совпадающего элемента или
            ``None``, если опорный текст или подходящий элемент не найден.
        """
        # Шаг 1 — найти позицию опорного текста через OCR
        anchor_result = ocr_engine.find_text(screenshot, anchor_text)
        if anchor_result is None:
            return None
        anchor_cx, anchor_cy = anchor_result.bbox.center

        # Шаг 2 — обнаружить элементы запрошенного класса
        candidates = self.find_by_class(screenshot, element_class)
        if not candidates:
            return None

        # Шаг 3 — вернуть ближайший по евклидовому расстоянию
        def _dist(det: DetectionResult) -> float:
            ex, ey = det.center
            return math.sqrt((ex - anchor_cx) ** 2 + (ey - anchor_cy) ** 2)

        return min(candidates, key=_dist)

    # ------------------------------------------------------------------
    # Реализация протокола Finder
    # ------------------------------------------------------------------

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Вернуть лучшее обнаружение в виде :class:`FindResult`.

        Реализует протокол :class:`~screenwalker.vision.screen_state.Finder`,
        чтобы движок мог использовать этот детектор взаимозаменяемо с
        OCR- и template-поисковиками.

        Args:
            image: Скриншот для поиска.
            query: Метка UI-класса для поиска (например, ``"button"``).
            threshold: Переопределить :attr:`confidence_threshold` для этого вызова.
            region: Необязательная область поиска.

        Returns:
            :class:`FindResult` для лучшего обнаружения или ``None``.
        """
        cutoff = threshold if threshold is not None else self.confidence_threshold
        results = self.find_by_class(image, query, region=region)
        if not results:
            return None
        best = results[0]  # уже отсортированы по уверенности
        if best.confidence < cutoff:
            return None
        return best.to_find_result()

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Вернуть все совпадающие обнаружения в виде объектов :class:`FindResult`.

        Args:
            image: Скриншот для поиска.
            query: Метка UI-класса.
            threshold: Переопределить :attr:`confidence_threshold` для этого вызова.
            region: Необязательная область поиска.

        Returns:
            Список :class:`FindResult`, отсортированный по убыванию уверенности.
        """
        cutoff = threshold if threshold is not None else self.confidence_threshold
        results = self.find_by_class(image, query, region=region)
        return [
            d.to_find_result()
            for d in results
            if d.confidence >= cutoff
        ]

    def detect_all(self, image: Image.Image) -> list[DetectionResult]:
        """Обнаружить все UI-элементы, видимые на *image*, независимо от класса.

        Полезно для исследовательской автоматизации и идентификации состояния экрана.

        Args:
            image: Скриншот для анализа.

        Returns:
            Все обнаруженные элементы, отсортированные по убыванию уверенности.
        """
        return self.detect(image)


# ---------------------------------------------------------------------------
# NullDetector — заглушка на случай отсутствия модели
# ---------------------------------------------------------------------------


class NullDetector:
    """Детектор-заглушка, всегда возвращающий пустые результаты.

    Используется, если модель не настроена или ``ultralytics`` не установлен.
    Реализует тот же интерфейс, что и :class:`UIElementDetector`.

    Attributes:
        available: Всегда ``False``.
        class_names: Всегда пустой.
    """

    available: bool = False
    class_names: dict[int, str] = {}

    def detect(self, screenshot: Image.Image, region: BBox | None = None) -> list[DetectionResult]:
        """Вернуть пустой список (модель не загружена)."""
        return []

    def find_by_class(
        self,
        screenshot: Image.Image,
        element_class: str,
        region: BBox | None = None,
    ) -> list[DetectionResult]:
        """Вернуть пустой список (модель не загружена)."""
        return []

    def find_nearest(
        self,
        screenshot: Image.Image,
        element_class: str,
        anchor_text: str,
        ocr_engine: Any,
    ) -> DetectionResult | None:
        """Вернуть ``None`` (модель не загружена)."""
        return None

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Вернуть ``None`` (модель не загружена)."""
        return None

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Вернуть пустой список (модель не загружена)."""
        return []

    def detect_all(self, image: Image.Image) -> list[DetectionResult]:
        """Вернуть пустой список (модель не загружена)."""
        return []


# ---------------------------------------------------------------------------
# Фабрика
# ---------------------------------------------------------------------------


def create_detector(config: Any) -> UIElementDetector | NullDetector:
    """Создать детектор на основе :class:`~screenwalker.utils.config.VisionConfig`.

    Логика выбора:

    1. Если ``config.yolo_enabled`` равно ``False`` → вернуть :class:`NullDetector`.
    2. Если ``config.yolo_model_path`` задан и файл существует →
       вернуть :class:`UIElementDetector` с этим путём.
    3. Если путь по умолчанию к OmniParser
       (``models/icon_detect/best.pt``) существует → использовать его.
    4. Иначе → вернуть :class:`NullDetector` и записать предупреждение в лог.

    Args:
        config: Экземпляр :class:`~screenwalker.utils.config.VisionConfig` (или любой
            объект с атрибутами ``yolo_enabled``, ``yolo_model_path``,
            ``yolo_confidence``).

    Returns:
        Готовый к использованию детектор (или :class:`NullDetector`, если недоступен).
    """
    import structlog as _sl
    _log = _sl.get_logger(__name__)

    if not getattr(config, "yolo_enabled", False):
        return NullDetector()

    confidence = getattr(config, "yolo_confidence", 0.50)

    # Явный путь к модели из конфига
    model_path_raw: str | None = getattr(config, "yolo_model_path", None)
    if model_path_raw:
        model_path = Path(model_path_raw)
        if model_path.exists():
            try:
                return UIElementDetector(
                    model_path=model_path,
                    confidence_threshold=confidence,
                )
            except (ImportError, FileNotFoundError) as exc:
                _log.warning("YOLO detector init failed", error=str(exc))
                return NullDetector()
        else:
            _log.warning("YOLO model not found", path=str(model_path))

    # Путь по умолчанию для OmniParser
    default_path = Path("models/icon_detect/best.pt")
    if default_path.exists():
        try:
            return UIElementDetector(
                model_path=default_path,
                confidence_threshold=confidence,
            )
        except (ImportError, FileNotFoundError) as exc:
            _log.warning("YOLO detector init failed (default path)", error=str(exc))

    _log.info(
        "YOLO enabled but no model found — using NullDetector. "
        "Run: python scripts/download_model.py"
    )
    return NullDetector()


# ---------------------------------------------------------------------------
# Псевдоним для обратной совместимости
# ---------------------------------------------------------------------------

#: Псевдоним для существующих мест вызова, использующих имя YOLODetector.
YOLODetector = UIElementDetector
