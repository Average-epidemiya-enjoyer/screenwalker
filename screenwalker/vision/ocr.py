"""OCR-модуль с подключаемой абстракцией движка.

Движок по умолчанию — Tesseract (через ``pytesseract``).
PaddleOCR доступен как опциональная альтернатива при установке
``config.vision.ocr_engine = "paddleocr"``.

Все движки реализуют Protocol :class:`OCREngine`. Результат OCR на уровне
слов представлен классом :class:`OCRResult`; высокоуровневый
:class:`~screenwalker.vision.screen_state.FindResult` возвращается методами
``find_text`` / ``find_all_text``, чтобы остальная система оставалась
независимой от конкретного движка.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PIL import Image
from rapidfuzz import fuzz as _rfuzz

from screenwalker.vision.screen_state import BBox, FindResult

if TYPE_CHECKING:
    from screenwalker.utils.config import AppConfig

# ---------------------------------------------------------------------------
# Опциональные тяжёлые зависимости — ошибка возникает при *вызове*, а не при
# импорте, чтобы модуль оставался импортируемым даже без установленных бинарных
# файлов. Имена на уровне модуля доступны для подмены в юнит-тестах.
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Тип результата OCR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRResult:
    """Результат OCR на уровне слова или строки.

    Attributes:
        text: Распознанная текстовая строка.
        confidence: Нормализованный показатель уверенности в диапазоне [0.0, 1.0].
        bbox: Ограничивающий прямоугольник текста в координатах изображения.
        raw_data: Метаданные, специфичные для движка (уровень/блок/строка/слово Tesseract,
            полигон PaddleOCR и т.д.).
    """

    text: str
    confidence: float
    bbox: BBox
    raw_data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol (структурный интерфейс)
# ---------------------------------------------------------------------------


class OCREngine(Protocol):
    """Структурный Protocol для реализаций OCR-движка."""

    def recognize(
        self, image: Image.Image, lang: str = "rus+eng"
    ) -> list[OCRResult]:
        """Запустить OCR на *image* и вернуть результаты на уровне слов.

        Args:
            image: Входное PIL-изображение.
            lang: Языковая подсказка (формат зависит от движка).

        Returns:
            Список :class:`OCRResult`, отфильтрованный по порогу уверенности.
        """
        ...

    def extract_text(self, image: Image.Image) -> str:
        """Извлечь весь распознанный текст в виде единой строки, объединённой пробелами.

        Args:
            image: Входное PIL-изображение.

        Returns:
            Извлечённая текстовая строка.
        """
        ...

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Найти текст *query* внутри *image*.

        Args:
            image: Скриншот для поиска.
            query: Искомая текстовая строка.
            threshold: Минимальный показатель нечёткого совпадения в диапазоне [0.0, 1.0].
            region: Опциональный ограничивающий прямоугольник для ограничения области поиска.

        Returns:
            Лучший :class:`FindResult` выше *threshold*, или None.
        """
        ...

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Найти все вхождения *query* в *image*.

        Args:
            image: Скриншот для поиска.
            query: Искомая текстовая строка.
            threshold: Минимальный показатель уверенности.
            region: Опциональная область поиска.

        Returns:
            Список :class:`FindResult`, отсортированный по убыванию уверенности.
        """
        ...


# ---------------------------------------------------------------------------
# Движок Tesseract
# ---------------------------------------------------------------------------


class TesseractEngine:
    """OCR-движок на основе Tesseract через pytesseract.

    Пайплайн предобработки (применяется при *preprocess* = True):
    1. Преобразование в оттенки серого.
    2. Опциональное удвоение разрешения (*resize*) для мелкого текста.
    3. Бинаризация через ``cv2.adaptiveThreshold``.
    4. Опциональное шумоподавление через ``cv2.fastNlMeansDenoising`` (*denoise*).

    Attributes:
        lang: Языковой код(ы) Tesseract (например ``"eng"`` или ``"eng+rus"``).
        config: Дополнительная строка конфигурации Tesseract (например ``"--psm 6"``).
        preprocess: Применять пайплайн предобработки перед OCR.
        confidence_threshold: Минимальный показатель уверенности слова Tesseract (0–100).
            Слова ниже этого значения отбрасываются. По умолчанию: 60.
        resize: Удвоить разрешение изображения перед OCR (помогает с мелким текстом).
        denoise: Применить нелокальное среднее шумоподавление перед OCR.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 6",
        preprocess: bool = True,
        confidence_threshold: int = 60,
        resize: bool = False,
        denoise: bool = False,
    ) -> None:
        self.lang = lang
        self.config = config
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.resize = resize
        self.denoise = denoise

    # ------------------------------------------------------------------
    # Предобработка
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Применить пайплайн предобработки к *image*.

        Args:
            image: Входное PIL-изображение (любой режим).

        Returns:
            Предобработанное PIL-изображение в режиме ``"L"`` (оттенки серого).

        Raises:
            ImportError: Если *opencv-python-headless* не установлен.
        """
        if cv2 is None:
            raise ImportError(
                "opencv-python-headless is required for OCR pre-processing: "
                "pip install opencv-python-headless"
            )

        # Шаг 1 — преобразование в оттенки серого
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Шаг 2 — опциональное масштабирование ×2 (бикубическая интерполяция для качества)
        if self.resize:
            h, w = gray.shape
            gray = cv2.resize(
                gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC
            )

        # Шаг 3 — адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Шаг 4 — опциональное шумоподавление
        if self.denoise:
            binary = cv2.fastNlMeansDenoising(binary, h=10)

        return Image.fromarray(binary)

    # ------------------------------------------------------------------
    # Основное распознавание
    # ------------------------------------------------------------------

    def recognize(
        self, image: Image.Image, lang: str | None = None
    ) -> list[OCRResult]:
        """Запустить Tesseract на *image* и вернуть отфильтрованные результаты на уровне слов.

        Args:
            image: Входное PIL-изображение.
            lang: Переопределить язык движка для данного вызова.

        Returns:
            Список :class:`OCRResult` с уверенностью >=
            :attr:`confidence_threshold` / 100.

        Raises:
            ImportError: Если *pytesseract* не установлен.
        """
        if pytesseract is None:
            raise ImportError(
                "pytesseract is required: pip install pytesseract"
            )

        effective_lang = lang if lang is not None else self.lang
        processed = self._preprocess(image) if self.preprocess else image

        data = pytesseract.image_to_data(
            processed,
            lang=effective_lang,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )

        results: list[OCRResult] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            try:
                conf = int(data["conf"][i])
            except (ValueError, TypeError):
                continue
            # conf == -1 для строк, не являющихся словами (заголовки блоков/строк/абзацев)
            if not text or conf < 0 or conf < self.confidence_threshold:
                continue

            bbox = BBox(
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
            )
            results.append(
                OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=bbox,
                    raw_data={
                        "level": data.get("level", [None] * n)[i],
                        "block_num": data.get("block_num", [None] * n)[i],
                        "line_num": data.get("line_num", [None] * n)[i],
                        "word_num": data.get("word_num", [None] * n)[i],
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # Группировка по строкам
    # ------------------------------------------------------------------

    def _group_into_lines(
        self,
        results: list[OCRResult],
    ) -> list[OCRResult]:
        """Объединить результаты на уровне слов в результаты на уровне строк.

        Два слова считаются находящимися на одной строке, если расстояние
        между вертикальными центрами не превышает половины высоты более
        высокого из слов.

        Args:
            results: Список :class:`OCRResult` на уровне слов (из :meth:`recognize`).

        Returns:
            Список :class:`OCRResult` на уровне строк, отсортированный сверху вниз,
            слева направо.
        """
        if not results:
            return []

        # Сортировка по (y_center, x), чтобы слова шли слева направо в пределах строки
        def _cy(r: OCRResult) -> int:
            return r.bbox.y + r.bbox.h // 2

        sorted_words = sorted(results, key=lambda r: (_cy(r), r.bbox.x))

        lines: list[list[OCRResult]] = []
        current_line: list[OCRResult] = [sorted_words[0]]

        for word in sorted_words[1:]:
            last = current_line[-1]
            tol = max(last.bbox.h, word.bbox.h) // 2
            if abs(_cy(word) - _cy(last)) <= tol:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        merged: list[OCRResult] = []
        for line_words in lines:
            text = " ".join(w.text for w in line_words)
            avg_conf = sum(w.confidence for w in line_words) / len(line_words)
            min_x = min(w.bbox.x for w in line_words)
            min_y = min(w.bbox.y for w in line_words)
            max_x = max(w.bbox.x + w.bbox.w for w in line_words)
            max_y = max(w.bbox.y + w.bbox.h for w in line_words)
            merged.append(
                OCRResult(
                    text=text,
                    confidence=avg_conf,
                    bbox=BBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y),
                    raw_data={},
                )
            )
        return merged

    # ------------------------------------------------------------------
    # Высокоуровневые методы поиска
    # ------------------------------------------------------------------

    def extract_text(self, image: Image.Image) -> str:
        """Извлечь весь текст из *image* в виде строки, объединённой пробелами.

        Args:
            image: Входное PIL-изображение.

        Returns:
            Распознанный текст, слова объединены пробелами.
        """
        return " ".join(r.text for r in self.recognize(image))

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> FindResult | None:
        """Найти наилучшее вхождение *query* в *image*.

        Args:
            image: Скриншот для поиска.
            query: Искомый текст.
            threshold: Минимальный показатель совпадения в диапазоне [0.0, 1.0].
            region: Опциональный ограничивающий прямоугольник для ограничения области поиска.
            fuzzy: Использовать ``partial_ratio`` из RapidFuzz при True; точную
                проверку подстроки при False.
            synonym_registry: Опциональный :class:`~screenwalker.matching.SynonymRegistry`
                для расширения *query* при отсутствии прямых результатов.

        Returns:
            :class:`FindResult` для наилучшего совпадения, или None.
        """
        matches = self.find_all_text(
            image, query, threshold=threshold, region=region, fuzzy=fuzzy,
            synonym_registry=synonym_registry,
        )
        return matches[0] if matches else None

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> list[FindResult]:
        """Найти все вхождения *query* в *image*.

        Если *synonym_registry* указан и прямой поиск не дал результатов,
        запрос расширяется до всех известных псевдонимов и каждый из них
        ищется по очереди.

        Args:
            image: Скриншот для поиска.
            query: Искомый текст.
            threshold: Минимальный показатель уверенности в диапазоне [0.0, 1.0].
            region: Опциональный ограничивающий прямоугольник для ограничения области поиска.
            fuzzy: Использовать нечёткое сопоставление при True.
            synonym_registry: Опциональный реестр для резервного расширения синонимов.

        Returns:
            Список :class:`FindResult`, отсортированный по убыванию уверенности.
        """
        search_image = image
        offset_x, offset_y = 0, 0
        if region is not None:
            search_image = image.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            offset_x, offset_y = region.x, region.y

        word_results = self.recognize(search_image)
        line_results = self._group_into_lines(word_results)

        def _score_lines(q: str) -> list[FindResult]:
            q_norm = q.lower().strip()
            found: list[FindResult] = []
            for result in line_results:
                text_norm = result.text.lower().strip()
                if not text_norm:
                    continue
                if fuzzy:
                    score = _rfuzz.partial_ratio(q_norm, text_norm) / 100.0
                else:
                    score = 1.0 if q_norm in text_norm else 0.0
                if score >= threshold:
                    final_bbox = result.bbox.offset(offset_x, offset_y)
                    found.append(
                        FindResult(
                            element=result.text,
                            confidence=score,
                            bbox=final_bbox,
                            method="ocr",
                            metadata={"ocr_confidence": result.confidence},
                        )
                    )
            return found

        matches = _score_lines(query)

        if not matches and synonym_registry is not None:
            aliases = synonym_registry.all_aliases(query) - {query.lower().strip()}
            for alias in aliases:
                alias_matches = _score_lines(alias)
                matches.extend(alias_matches)

        return sorted(matches, key=lambda r: r.confidence, reverse=True)


# ---------------------------------------------------------------------------
# Движок PaddleOCR (опциональный)
# ---------------------------------------------------------------------------


class PaddleOCREngine:
    """Опциональный OCR-движок на основе PaddleOCR.

    Требует дополнительных зависимостей ``paddleocr``: ``pip install screenwalker[paddleocr]``.

    Attributes:
        lang: Языковой код PaddleOCR (например ``"ru"``, ``"en"``, ``"ch"``).
    """

    def __init__(self, lang: str = "ru") -> None:
        """Инициализировать PaddleOCREngine.

        Args:
            lang: Языковой код PaddleOCR.

        Raises:
            ImportError: Если paddleocr не установлен.
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install with: pip install screenwalker[paddleocr]"
            ) from exc

        self.lang = lang
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def recognize(
        self, image: Image.Image, lang: str | None = None
    ) -> list[OCRResult]:
        """Запустить PaddleOCR на *image*.

        Args:
            image: Входное PIL-изображение.
            lang: Игнорируется — язык PaddleOCR задаётся при создании объекта.

        Returns:
            Список :class:`OCRResult`, по одному на каждый обнаруженный текстовый блок.
        """
        img_array = np.array(image.convert("RGB"))
        raw = self._ocr.ocr(img_array, cls=True)

        results: list[OCRResult] = []
        # raw имеет вид [[строка, ...]], где строка = [точки_рамки, (текст, уверенность)]
        page = raw[0] if raw else []
        for item in page:
            box, (text, conf) = item
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs)) - x
            h = int(max(ys)) - y
            results.append(
                OCRResult(
                    text=str(text),
                    confidence=float(conf),
                    bbox=BBox(x=x, y=y, w=w, h=h),
                    raw_data={"box": box},
                )
            )
        return results

    def extract_text(self, image: Image.Image) -> str:
        """Извлечь весь текст из *image* в виде строки, объединённой пробелами."""
        return " ".join(r.text for r in self.recognize(image))

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> FindResult | None:
        """Найти наилучшее вхождение *query* в *image*."""
        matches = self.find_all_text(
            image, query, threshold=threshold, region=region, fuzzy=fuzzy,
            synonym_registry=synonym_registry,
        )
        return matches[0] if matches else None

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> list[FindResult]:
        """Найти все вхождения *query* в *image*."""
        search_image = image
        offset_x, offset_y = 0, 0
        if region is not None:
            search_image = image.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            offset_x, offset_y = region.x, region.y

        ocr_results = self.recognize(search_image)

        def _score_results(q: str) -> list[FindResult]:
            q_norm = q.lower().strip()
            found: list[FindResult] = []
            for result in ocr_results:
                text_norm = result.text.lower().strip()
                if not text_norm:
                    continue
                if fuzzy:
                    score = _rfuzz.partial_ratio(q_norm, text_norm) / 100.0
                else:
                    score = 1.0 if q_norm in text_norm else 0.0
                if score >= threshold:
                    final_bbox = result.bbox.offset(offset_x, offset_y)
                    found.append(
                        FindResult(
                            element=result.text,
                            confidence=score,
                            bbox=final_bbox,
                            method="ocr",
                            metadata={"ocr_confidence": result.confidence},
                        )
                    )
            return found

        matches = _score_results(query)

        if not matches and synonym_registry is not None:
            aliases = synonym_registry.all_aliases(query) - {query.lower().strip()}
            for alias in aliases:
                alias_matches = _score_results(alias)
                matches.extend(alias_matches)

        return sorted(matches, key=lambda r: r.confidence, reverse=True)


# ---------------------------------------------------------------------------
# Фабрики
# ---------------------------------------------------------------------------


def create_ocr_engine(config: AppConfig) -> OCREngine:
    """Создать OCR-движок из :class:`~screenwalker.utils.config.AppConfig`.

    Args:
        config: Конфигурация приложения.

    Returns:
        Настроенный экземпляр :class:`OCREngine`.

    Raises:
        ValueError: Если ``config.vision.ocr_engine`` не распознан.
    """
    name = config.vision.ocr_engine
    if name == "tesseract":
        return TesseractEngine(
            lang=config.vision.ocr_lang,
            config=config.vision.ocr_config,
            preprocess=config.vision.ocr_preprocess,
        )
    if name == "paddleocr":
        return PaddleOCREngine(lang=config.vision.ocr_lang)
    raise ValueError(
        f"Unknown OCR engine: {name!r}. Available: ['tesseract', 'paddleocr']"
    )


def build_ocr_engine(engine_name: str = "tesseract", **kwargs: object) -> OCREngine:
    """Фабрика для создания OCR-движка по имени.

    Это низкоуровневая фабрика; предпочтительно использовать :func:`create_ocr_engine`,
    если доступен :class:`~screenwalker.utils.config.AppConfig`.

    Args:
        engine_name: Идентификатор движка — ``"tesseract"`` или ``"paddleocr"``.
        **kwargs: Передаются в конструктор движка.

    Returns:
        Экземпляр движка, совместимый с :class:`OCREngine`.

    Raises:
        ValueError: Если *engine_name* не распознан.
    """
    engines = {
        "tesseract": TesseractEngine,
        "paddleocr": PaddleOCREngine,
    }
    cls = engines.get(engine_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown OCR engine: {engine_name!r}. "
            f"Available: {list(engines)}"
        )
    return cls(**kwargs)  # type: ignore[return-value]
