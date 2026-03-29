"""Идентификация состояния экрана и унифицированный тип FindResult.

:class:`FindResult` — единственная структура данных, возвращаемая каждым методом
компьютерного зрения (OCR, template matching, YOLO). :class:`ScreenStateDetector`
объединяет несколько методов поиска для идентификации текущего «логического экрана» UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from PIL import Image
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from screenwalker.vision.ocr import OCRResult

log = structlog.get_logger(__name__)


# ── Общая геометрия ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BBox:
    """Ограничивающий прямоугольник с выравниванием по осям в координатах экрана.

    Attributes:
        x: Левый край (пикселей от левого края экрана/области).
        y: Верхний край (пикселей от верхнего края экрана/области).
        w: Ширина в пикселях.
        h: Высота в пикселях.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        """Пиксельные координаты центра ограничивающего прямоугольника."""
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def right(self) -> int:
        """Координата правого края в пикселях."""
        return self.x + self.w

    @property
    def bottom(self) -> int:
        """Координата нижнего края в пикселях."""
        return self.y + self.h

    def offset(self, dx: int, dy: int) -> BBox:
        """Возвращает новый BBox, сдвинутый на (dx, dy).

        Args:
            dx: Горизонтальное смещение в пикселях.
            dy: Вертикальное смещение в пикселях.

        Returns:
            Новый смещённый BBox (исходный не изменяется).
        """
        return BBox(self.x + dx, self.y + dy, self.w, self.h)


# ── Унифицированный результат поиска ─────────────────────────────────────────


@dataclass(frozen=True)
class FindResult:
    """Унифицированный результат, возвращаемый каждым методом поиска.

    Attributes:
        element: Читаемое описание найденного элемента (текст, метка и т.д.).
        confidence: Оценка уверенности в диапазоне [0.0, 1.0].
        bbox: Ограничивающий прямоугольник найденного элемента в координатах экрана.
        method: Метод компьютерного зрения, давший результат (``ocr``, ``template``, ``yolo``).
        metadata: Дополнительные данные (например, сырые OCR-боксы, идентификатор класса YOLO).
    """

    element: str
    confidence: float
    bbox: BBox
    method: str
    metadata: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Удобный доступ к bbox.center."""
        return self.bbox.center


# ── Протокол поисковика ────────────────────────────────────────────────────────


class Finder(Protocol):
    """Протокол, который должны реализовывать все методы поиска.

    Любой класс, реализующий этот протокол, может взаимозаменяемо использоваться
    движком для поиска UI-элементов.
    """

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.80,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Пытается найти *query* в *image*.

        Args:
            image: Скриншот (полный экран или обрезанная область).
            query: Поисковый запрос — текст для OCR, путь к изображению для шаблона.
            threshold: Минимальный порог уверенности для принятия совпадения.
            region: Опциональный ограничивающий прямоугольник для сужения поиска.

        Returns:
            :class:`FindResult` если найдено выше порога, иначе None.
        """
        ...

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.80,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Находит все вхождения *query* в *image*.

        Args:
            image: Скриншот для поиска.
            query: Поисковый запрос.
            threshold: Минимальный порог уверенности.
            region: Опциональный ограничивающий прямоугольник для сужения поиска.

        Returns:
            Список :class:`FindResult`, отсортированных по убыванию уверенности.
        """
        ...


# ── Отпечаток экрана (Pydantic, сериализуемый в YAML) ─────────────────────────


class ScreenFingerprint(BaseModel):
    """Декларативный дескриптор именованного логического экрана.

    Каждый отпечаток сопоставляется с текущим скриншотом во время выполнения.
    Тексты сравниваются нечётким сопоставлением; шаблоны — OpenCV template matching.

    Attributes:
        screen_id: Уникальный идентификатор экрана (например, ``"notepad_main"``).
        required_texts: Тексты, которые ДОЛЖНЫ присутствовать. Отсутствие любого
            снижает оценку на ``-1.0``.
        forbidden_texts: Тексты, которые НЕ должны присутствовать. Наличие любого
            снижает оценку на ``-1.0``.
        required_templates: Имена шаблонов, которые должны быть видны.
            Каждое совпадение добавляет ``+0.5`` к оценке.
        optional_texts: Тексты, наличие которых добавляет ``+0.3`` (только бонус).
        match_threshold: Минимальная нормализованная уверенность для принятия совпадения.
    """

    screen_id: str
    required_texts: list[str] = Field(default_factory=list)
    forbidden_texts: list[str] = Field(default_factory=list)
    required_templates: list[str] = Field(default_factory=list)
    optional_texts: list[str] = Field(default_factory=list)
    match_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


# ── Результат идентификации ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenIdentification:
    """Результат :meth:`ScreenStateAnalyzer.identify`.

    Attributes:
        screen_id: Идентификатор наиболее подходящего отпечатка или ``None``.
        confidence: Нормализованная оценка в диапазоне [0.0, 1.0].
        matched_texts: Обязательные/опциональные тексты, которые были найдены.
        matched_templates: Имена шаблонов, которые были найдены.
        ocr_results: Сырые результаты OCR, использованные при идентификации.
        is_popup: True если поверх идентифицированного экрана обнаружено всплывающее окно.
    """

    screen_id: str | None
    confidence: float
    matched_texts: list[str] = field(default_factory=list)
    matched_templates: list[str] = field(default_factory=list)
    ocr_results: list[Any] = field(default_factory=list)
    is_popup: bool = False


# ── Информация о попапе ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PopupInfo:
    """Результат :meth:`ScreenStateAnalyzer.detect_popup`.

    Attributes:
        popup_type: Один из вариантов: ``"error"``, ``"confirm"``, ``"loading"``.
        confidence: Нормализованная оценка совпадения.
        matched_texts: Тексты попапа, которые были найдены.
    """

    popup_type: str
    confidence: float
    matched_texts: list[str] = field(default_factory=list)


# ── Встроенные отпечатки попапов ──────────────────────────────────────────────

_POPUP_FINGERPRINTS: list[ScreenFingerprint] = [
    ScreenFingerprint(
        screen_id="error_popup",
        required_texts=["Error"],
        optional_texts=["OK", "Close", "Cancel", "failed", "cannot", "unable"],
        match_threshold=0.5,
    ),
    ScreenFingerprint(
        screen_id="confirm_popup",
        required_texts=["OK", "Cancel"],
        optional_texts=["Yes", "No", "Confirm", "Are you sure"],
        match_threshold=0.6,
    ),
    ScreenFingerprint(
        screen_id="loading_popup",
        required_texts=["Please wait"],
        optional_texts=["Loading", "Processing", "…", "...", "progress"],
        match_threshold=0.4,
    ),
]


# ── Основной анализатор ───────────────────────────────────────────────────────


class ScreenStateAnalyzer:
    """Идентифицирует текущий логический экран путём оценки зарегистрированных отпечатков.

    Анализатор запускает OCR на скриншоте один раз и переиспользует результаты
    при проверке всех отпечатков. Template matching выполняется только при наличии
    :class:`~screenwalker.vision.template_match.TemplateMatcher`.

    Оценка для каждого отпечатка:

    * Обязательный текст найден    → ``+1.0``
    * Обязательный текст отсутствует → ``-1.0`` (сырая оценка может быть отрицательной)
    * Запрещённый текст найден     → ``-1.0``
    * Обязательный шаблон найден   → ``+0.5``
    * Опциональный текст найден    → ``+0.3``

    Сырая оценка нормализуется относительно максимально достижимой для данного
    отпечатка. Отпечаток с наибольшей нормализованной оценкой выше своего
    ``match_threshold`` побеждает.

    Args:
        fingerprints: Список объектов :class:`ScreenFingerprint` для сопоставления.
        ocr_engine: Экземпляр OCR-движка (должен реализовывать ``recognize()``).
            Передайте ``None`` для отключения OCR-сопоставления.
        template_matcher: Экземпляр :class:`~screenwalker.vision.template_match.TemplateMatcher`.
            Передайте ``None`` для отключения template matching.
        fuzzy_threshold: Минимальная оценка ``rapidfuzz.partial_ratio`` (0–100) при
            сравнении OCR-текстов с текстами отпечатков.
    """

    def __init__(
        self,
        fingerprints: list[ScreenFingerprint] | None = None,
        ocr_engine: Any = None,
        template_matcher: Any = None,
        fuzzy_threshold: int = 70,
        synonym_registry: Any | None = None,
    ) -> None:
        self._fingerprints: list[ScreenFingerprint] = fingerprints or []
        self._ocr = ocr_engine
        self._matcher = template_matcher
        self._fuzzy_threshold = fuzzy_threshold
        self._synonym_registry = synonym_registry

    # ── Публичный API ─────────────────────────────────────────────────────────

    def identify(self, screenshot: Image.Image) -> ScreenIdentification:
        """Идентифицирует текущий экран по скриншоту.

        Args:
            screenshot: Скриншот полного экрана или окна.

        Returns:
            :class:`ScreenIdentification` для наиболее подходящего отпечатка.
            Если ни один отпечаток не превышает своего порога, ``screen_id`` равен ``None``,
            а ``confidence`` равен ``0.0``.
        """
        ocr_results = self._run_ocr(screenshot)
        ocr_texts = self._collect_texts(ocr_results)

        best_id: str | None = None
        best_conf: float = 0.0
        best_texts: list[str] = []
        best_templates: list[str] = []

        for fp in self._fingerprints:
            conf, texts, templates = self._score(fp, screenshot, ocr_texts)
            log.debug(
                "screen_state.scored",
                screen_id=fp.screen_id,
                confidence=round(conf, 3),
            )
            if conf >= fp.match_threshold and conf > best_conf:
                best_id = fp.screen_id
                best_conf = conf
                best_texts = texts
                best_templates = templates

        popup = self.detect_popup(screenshot, ocr_results=ocr_results)

        return ScreenIdentification(
            screen_id=best_id,
            confidence=best_conf,
            matched_texts=best_texts,
            matched_templates=best_templates,
            ocr_results=ocr_results,
            is_popup=popup is not None,
        )

    def verify(self, screenshot: Image.Image, expected_screen_id: str) -> bool:
        """Проверяет, соответствует ли скриншот конкретному экрану.

        Args:
            screenshot: Скриншот для проверки.
            expected_screen_id: ``screen_id`` для сопоставления.

        Returns:
            ``True`` если идентифицированный экран совпадает с *expected_screen_id*.
        """
        result = self.identify(screenshot)
        return result.screen_id == expected_screen_id

    def detect_popup(
        self,
        screenshot: Image.Image,
        *,
        ocr_results: list[Any] | None = None,
    ) -> PopupInfo | None:
        """Определяет, отображается ли в данный момент всплывающее диалоговое окно.

        Проверяет встроенные отпечатки попапов (error, confirm, loading).

        Args:
            screenshot: Скриншот для анализа.
            ocr_results: Предварительно вычисленные результаты OCR для повторного
                использования (избегает второго прохода OCR при вызове из :meth:`identify`).

        Returns:
            :class:`PopupInfo` для наиболее подходящего типа попапа или ``None``
            если попап не обнаружен.
        """
        if ocr_results is None:
            ocr_results = self._run_ocr(screenshot)
        ocr_texts = self._collect_texts(ocr_results)

        best_popup: PopupInfo | None = None
        best_conf: float = 0.0

        for fp in _POPUP_FINGERPRINTS:
            conf, texts, _ = self._score(fp, screenshot, ocr_texts)
            if conf >= fp.match_threshold and conf > best_conf:
                best_conf = conf
                popup_type = fp.screen_id.replace("_popup", "")
                best_popup = PopupInfo(
                    popup_type=popup_type,
                    confidence=conf,
                    matched_texts=texts,
                )

        return best_popup

    # ── Внутренние вспомогательные методы ────────────────────────────────────

    def _run_ocr(self, screenshot: Image.Image) -> list[Any]:
        """Запускает OCR и возвращает пословные результаты (пустой список при отсутствии движка)."""
        if self._ocr is None:
            return []
        try:
            return self._ocr.recognize(screenshot)
        except Exception:
            log.warning("screen_state.ocr_failed", exc_info=True)
            return []

    def _collect_texts(self, ocr_results: list[Any]) -> list[str]:
        """Строит плоский список строк для поиска из результатов OCR.

        Включает тексты отдельных слов, сгруппированные по строкам тексты и
        полную конкатенированную строку — позволяет сопоставлять как на уровне
        слов, так и на уровне фраз.
        """
        if not ocr_results:
            return []

        word_texts = [r.text for r in ocr_results if r.text.strip()]
        all_texts: list[str] = list(word_texts)

        # Добавляем тексты, сгруппированные по строкам, если OCR-движок поддерживает группировку
        if self._ocr is not None and hasattr(self._ocr, "_group_into_lines"):
            try:
                line_results = self._ocr._group_into_lines(ocr_results)
                for r in line_results:
                    if r.text.strip() and r.text not in all_texts:
                        all_texts.append(r.text)
            except Exception:
                log.warning("screen_state.line_grouping_failed", exc_info=True)

        full = " ".join(word_texts)
        if full and full not in all_texts:
            all_texts.append(full)

        return all_texts

    def _text_found(self, query: str, ocr_texts: list[str]) -> bool:
        """Возвращает True если *query* (или любой его синоним) найден в *ocr_texts*."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            # Точное подстрочное совпадение как запасной вариант
            q = query.lower()
            if any(q in t.lower() for t in ocr_texts):
                return True
            if self._synonym_registry is not None:
                for alias in self._synonym_registry.all_aliases(query):
                    if any(alias in t.lower() for t in ocr_texts):
                        return True
            return False

        threshold = self._fuzzy_threshold
        for text in ocr_texts:
            if fuzz.partial_ratio(query.lower(), text.lower()) >= threshold:
                return True

        if self._synonym_registry is not None:
            aliases = self._synonym_registry.all_aliases(query) - {query.lower().strip()}
            for alias in aliases:
                for text in ocr_texts:
                    if fuzz.partial_ratio(alias, text.lower()) >= threshold:
                        return True

        return False

    def _score(
        self,
        fp: ScreenFingerprint,
        screenshot: Image.Image,
        ocr_texts: list[str],
    ) -> tuple[float, list[str], list[str]]:
        """Оценивает *fp* относительно текущего скриншота и текстов OCR.

        Returns:
            Кортеж (нормализованная_уверенность, совпавшие_тексты, совпавшие_шаблоны).
        """
        raw: float = 0.0
        matched_texts: list[str] = []
        matched_templates: list[str] = []

        for text in fp.required_texts:
            if self._text_found(text, ocr_texts):
                raw += 1.0
                matched_texts.append(text)
            else:
                raw -= 1.0

        for text in fp.forbidden_texts:
            if self._text_found(text, ocr_texts):
                raw -= 1.0

        if self._matcher is not None:
            for tmpl in fp.required_templates:
                try:
                    result = self._matcher.find_one(screenshot, tmpl, threshold=0.75)
                    if result is not None:
                        raw += 0.5
                        matched_templates.append(tmpl)
                except Exception:
                    log.warning(
                        "screen_state.template_match_failed",
                        template=tmpl,
                        exc_info=True,
                    )

        for text in fp.optional_texts:
            if self._text_found(text, ocr_texts):
                raw += 0.3
                matched_texts.append(text)

        max_score = (
            len(fp.required_texts) * 1.0
            + len(fp.required_templates) * 0.5
            + len(fp.optional_texts) * 0.3
        )

        if max_score > 0:
            confidence = max(0.0, min(1.0, raw / max_score))
        else:
            confidence = 1.0 if raw >= 0 else 0.0

        return confidence, matched_texts, matched_templates


# ── Обратносовместимый детектор ───────────────────────────────────────────────


class ScreenStateDetector:
    """Идентифицирует текущий «логический экран», объединяя несколько методов поиска.

    Оборачивает :class:`ScreenStateAnalyzer` и предоставляет устаревший API
    ``register`` / ``identify`` / ``load_from_config`` для совместимости
    с существующими вызывающими местами.

    Args:
        ocr_engine: Опциональный OCR-движок для текстовых якорей.
        template_matcher: Опциональный template matcher для визуальных якорей.

    Example:
        >>> detector = ScreenStateDetector()
        >>> detector.register("login_screen", ocr_anchors=["Sign In"])
        >>> state = detector.identify(screenshot)
    """

    def __init__(
        self,
        ocr_engine: Any = None,
        template_matcher: Any = None,
        fuzzy_threshold: int = 70,
    ) -> None:
        self._ocr = ocr_engine
        self._matcher = template_matcher
        self._fuzzy_threshold = fuzzy_threshold
        self._fingerprints: list[ScreenFingerprint] = []
        self._analyzer: ScreenStateAnalyzer | None = None

    def _get_analyzer(self) -> ScreenStateAnalyzer:
        """Лениво создаёт внутренний :class:`ScreenStateAnalyzer`.

        Кэшированный экземпляр сбрасывается (устанавливается в ``None``) методами
        :meth:`register` и :meth:`load_from_config` при изменении списка отпечатков.
        """
        if self._analyzer is None:
            self._analyzer = ScreenStateAnalyzer(
                fingerprints=self._fingerprints,
                ocr_engine=self._ocr,
                template_matcher=self._matcher,
                fuzzy_threshold=self._fuzzy_threshold,
            )
        return self._analyzer

    def register(
        self,
        state_id: str,
        ocr_anchors: list[str] | None = None,
        template_anchors: list[str] | None = None,
        threshold: float = 0.75,
    ) -> None:
        """Регистрирует именованное состояние экрана с идентифицирующими якорями.

        Args:
            state_id: Уникальный идентификатор состояния экрана.
            ocr_anchors: Список текстовых строк, которые должны быть видны.
            template_anchors: Список путей к шаблонам, которые должны быть видны.
            threshold: Минимальная средняя уверенность для активации состояния.
        """
        fp = ScreenFingerprint(
            screen_id=state_id,
            required_texts=ocr_anchors or [],
            required_templates=template_anchors or [],
            match_threshold=threshold,
        )
        self._fingerprints.append(fp)
        self._analyzer = None  # сбросить кэшированный анализатор

    def identify(self, image: Image.Image) -> str | None:
        """Идентифицирует текущее состояние экрана по скриншоту.

        Args:
            image: Скриншот полного экрана или окна.

        Returns:
            Идентификатор наиболее подходящего зарегистрированного состояния
            или None если ни одно состояние не превышает своего порога.
        """
        result = self._get_analyzer().identify(image)
        return result.screen_id

    def load_from_config(self, states_config: dict) -> None:
        """Массово регистрирует состояния из конфигурационного словаря.

        Ожидаемый формат::

            {
                "login_screen": {
                    "required_texts": ["Sign In"],
                    "forbidden_texts": [],
                    "required_templates": [],
                    "optional_texts": ["Forgot Password"],
                    "match_threshold": 0.75
                },
                ...
            }

        Args:
            states_config: Словарь ``state_id`` → поля отпечатка.
        """
        for state_id, cfg in states_config.items():
            try:
                fp = ScreenFingerprint(
                    screen_id=state_id,
                    required_texts=cfg.get("required_texts", []),
                    forbidden_texts=cfg.get("forbidden_texts", []),
                    required_templates=cfg.get("required_templates", []),
                    optional_texts=cfg.get("optional_texts", []),
                    match_threshold=cfg.get("match_threshold", 0.75),
                )
            except Exception:
                log.warning(
                    "screen_state.invalid_fingerprint_skipped",
                    state_id=state_id,
                    exc_info=True,
                )
                continue
            self._fingerprints.append(fp)
        self._analyzer = None  # сбросить кэшированный анализатор
