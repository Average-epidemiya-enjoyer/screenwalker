"""Обнаружение и автоматическое закрытие всплывающих окон.

:class:`PopupHandler` ведёт реестр известных паттернов popup-окон, загружаемых из
YAML-файла конфигурации.  Перед каждым шагом движок вызывает
:meth:`PopupHandler.detect_and_handle` для закрытия любых диалогов-оверлеев,
чтобы они не блокировали основной поток автоматизации.

Пример конфигурации (``config/popups.yaml``)::

    popups:
      - id: error_dialog
        indicators: ["Error", "Something went wrong"]
        action: click_button
        button_text: ["OK", "Close"]

      - id: loading_screen
        indicators: ["Loading", "Please wait"]
        action: wait
        wait_timeout: 30
        wait_until_gone: true
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog
import yaml
from PIL import Image

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Модель данных
# ---------------------------------------------------------------------------


@dataclass
class PopupDefinition:
    """Описывает паттерн popup-окна, которое обработчик умеет обнаруживать и закрывать.

    Attributes:
        id: Уникальный идентификатор, используемый в сообщениях лога.
        indicators: OCR-строки, сигнализирующие о том, что popup видим.
            Уверенность = количество_совпавших / всего_индикаторов.
        templates: Необязательные имена шаблонных изображений для дополнительного
            визуального сопоставления.  Каждый совпавший шаблон добавляет +0.3
            к уверенности (максимум 1.0).
        action: Стратегия закрытия — ``"click_button"`` | ``"wait"`` | ``"close"``.
        button_text: Упорядоченный список надписей кнопок для клика
            (действие ``click_button``).
        wait_timeout: Максимальное время ожидания в секундах (действие ``wait``).
        wait_until_gone: Если True, опрашивать экран до тех пор, пока popup не исчезнет.
        priority: Popup-окна проверяются в порядке убывания приоритета.
        confidence_threshold: Минимальная доля совпавших индикаторов, при которой
            popup считается обнаруженным.
    """

    id: str
    indicators: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    action: str = "click_button"
    button_text: list[str] = field(default_factory=list)
    wait_timeout: float = 30.0
    wait_until_gone: bool = False
    priority: int = 0
    confidence_threshold: float = 0.4


@dataclass(frozen=True)
class PopupMatch:
    """Результат успешного обнаружения popup-окна.

    Attributes:
        definition: Совпавшее определение popup.
        confidence: Уверенность обнаружения в диапазоне ``[0.0, 1.0]``.
        matched_texts: Тексты-индикаторы, найденные на экране.
    """

    definition: PopupDefinition
    confidence: float
    matched_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PopupHandler
# ---------------------------------------------------------------------------


class PopupHandler:
    """Обнаруживает и закрывает диалоговые popup-окна перед выполнением шага.

    Attributes:
        definitions: Отсортированный список определений popup-окон (сначала
            наивысший приоритет).
    """

    def __init__(
        self,
        definitions: list[PopupDefinition],
        ocr_engine: Any,
        mouse: Any,
        template_matcher: Any = None,
    ) -> None:
        """Инициализировать со списком определений popup-окон.

        Args:
            definitions: Паттерны popup-окон для отслеживания.
            ocr_engine: Экземпляр OCR-движка (должен реализовывать ``recognize`` и
                ``find_text``).
            mouse: Контроллер мыши (должен реализовывать ``click(x, y)``).
            template_matcher: Необязательный экземпляр template-поисковика.
        """
        self.definitions: list[PopupDefinition] = sorted(
            definitions, key=lambda d: d.priority, reverse=True
        )
        self._ocr = ocr_engine
        self._mouse = mouse
        self._matcher = template_matcher

    # ------------------------------------------------------------------
    # Фабрика
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        ocr_engine: Any,
        mouse: Any,
        template_matcher: Any = None,
    ) -> "PopupHandler":
        """Загрузить определения popup-окон из YAML-файла.

        Args:
            path: Путь к YAML-файлу с корневым списком ``popups:``.
            ocr_engine: Экземпляр OCR-движка.
            mouse: Экземпляр контроллера мыши.
            template_matcher: Необязательный template-поисковик.

        Returns:
            Настроенный :class:`PopupHandler`.

        Raises:
            FileNotFoundError: Если *path* не существует.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Popup config not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        definitions: list[PopupDefinition] = []
        for entry in raw.get("popups", []):
            if not isinstance(entry, dict):
                continue
            try:
                definitions.append(PopupDefinition(
                    id=str(entry["id"]),
                    indicators=list(entry.get("indicators", [])),
                    templates=list(entry.get("templates", [])),
                    action=str(entry.get("action", "click_button")),
                    button_text=list(entry.get("button_text", [])),
                    wait_timeout=float(entry.get("wait_timeout", 30.0)),
                    wait_until_gone=bool(entry.get("wait_until_gone", False)),
                    priority=int(entry.get("priority", 0)),
                    confidence_threshold=float(entry.get("confidence_threshold", 0.4)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("popup_handler.invalid_definition", error=str(exc))

        return cls(definitions, ocr_engine, mouse, template_matcher)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def detect(self, screenshot: Image.Image) -> PopupMatch | None:
        """Определить, виден ли на экране какой-либо известный popup.

        Определения проверяются в порядке убывания приоритета.  Возвращается
        первое определение, чья уверенность достигает
        :attr:`~PopupDefinition.confidence_threshold`.

        Args:
            screenshot: Текущее изображение экрана.

        Returns:
            :class:`PopupMatch` для лучшего совпадения или ``None``, если popup
            не обнаружен.
        """
        ocr_texts = self._ocr_texts(screenshot)

        for defn in self.definitions:
            confidence, matched = self._score(defn, screenshot, ocr_texts)
            if confidence >= defn.confidence_threshold:
                log.debug(
                    "popup_handler.detected",
                    popup_id=defn.id,
                    confidence=round(confidence, 3),
                    matched=matched,
                )
                return PopupMatch(
                    definition=defn,
                    confidence=confidence,
                    matched_texts=matched,
                )

        return None

    def handle(
        self,
        screenshot: Image.Image,
        match: PopupMatch,
        capture_fn: Callable[[], Image.Image] | None = None,
    ) -> bool:
        """Закрыть обнаруженный popup.

        Args:
            screenshot: Текущее изображение экрана (используется для поиска кнопки).
            match: Результат совпадения popup, возвращённый методом :meth:`detect`.
            capture_fn: Необязательный вызываемый объект, возвращающий свежий
                скриншот.  Требуется для поведения ``wait_until_gone``.

        Returns:
            ``True`` если popup был успешно закрыт.
        """
        defn = match.definition
        log.info("popup_handler.handling", popup_id=defn.id, action=defn.action)

        if defn.action == "click_button":
            return self._handle_click_button(screenshot, defn)
        if defn.action == "close":
            return self._handle_close()
        if defn.action == "wait":
            return self._handle_wait(defn, capture_fn)

        log.warning("popup_handler.unknown_action", action=defn.action)
        return False

    def detect_and_handle(
        self,
        screenshot: Image.Image,
        capture_fn: Callable[[], Image.Image] | None = None,
    ) -> bool:
        """Обнаружить и закрыть popup за один вызов.

        Args:
            screenshot: Текущее изображение экрана.
            capture_fn: Необязательный вызываемый объект для получения свежих
                скриншотов (нужен стратегии ``wait_until_gone``).

        Returns:
            ``True`` если popup был найден и обработан, ``False`` если экран чист.
        """
        match = self.detect(screenshot)
        if match is None:
            return False
        return self.handle(screenshot, match, capture_fn=capture_fn)

    # ------------------------------------------------------------------
    # Стратегии закрытия
    # ------------------------------------------------------------------

    def _handle_click_button(self, screenshot: Image.Image, defn: PopupDefinition) -> bool:
        """Найти и кликнуть по первой доступной кнопке из списка."""
        if self._ocr is None:
            log.warning("popup_handler.no_ocr_engine")
            return False

        for btn_text in defn.button_text:
            try:
                result = self._ocr.find_text(screenshot, btn_text, threshold=0.50)
                if result is not None:
                    cx, cy = result.center
                    self._mouse.click(cx, cy)
                    log.info("popup_handler.button_clicked", button=btn_text, popup_id=defn.id)
                    time.sleep(0.3)
                    return True
            except Exception as exc:
                log.debug("popup_handler.button_not_found", button=btn_text, error=str(exc))

        log.warning(
            "popup_handler.no_button_found",
            popup_id=defn.id,
            tried=defn.button_text,
        )
        return False

    def _handle_close(self) -> bool:
        """Нажать Escape для закрытия popup."""
        try:
            import pyautogui
            pyautogui.press("escape")
            time.sleep(0.3)
            return True
        except Exception as exc:
            log.warning("popup_handler.escape_failed", error=str(exc))
            return False

    def _handle_wait(
        self,
        defn: PopupDefinition,
        capture_fn: Callable[[], Image.Image] | None,
    ) -> bool:
        """Ждать исчезновения popup или истечения таймаута."""
        if not defn.wait_until_gone or capture_fn is None:
            # Простое ожидание по времени
            time.sleep(min(defn.wait_timeout, 5.0))
            return True

        deadline = time.monotonic() + defn.wait_timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            fresh = capture_fn()
            if self.detect(fresh) is None:
                log.info("popup_handler.wait_resolved", popup_id=defn.id)
                return True

        log.warning(
            "popup_handler.wait_timeout",
            popup_id=defn.id,
            timeout=defn.wait_timeout,
        )
        return False

    # ------------------------------------------------------------------
    # Вспомогательные методы для подсчёта уверенности
    # ------------------------------------------------------------------

    def _ocr_texts(self, screenshot: Image.Image) -> list[str]:
        """Запустить OCR и вернуть плоский список распознанных строк."""
        if self._ocr is None:
            return []
        try:
            results = self._ocr.recognize(screenshot)
            words = [r.text for r in results if r.text.strip()]
            if words:
                words.append(" ".join(words))  # полная конкатенация для поиска фраз
            return words
        except Exception:
            return []

    def _score(
        self,
        defn: PopupDefinition,
        screenshot: Image.Image,
        ocr_texts: list[str],
    ) -> tuple[float, list[str]]:
        """Оценить *defn* относительно текущего содержимого экрана.

        Returns:
            ``(confidence, matched_indicators)``
        """
        try:
            from rapidfuzz import fuzz

            def _found(query: str) -> bool:
                return any(fuzz.partial_ratio(query.lower(), t.lower()) >= 70 for t in ocr_texts)
        except ImportError:
            def _found(query: str) -> bool:
                return any(query.lower() in t.lower() for t in ocr_texts)

        matched: list[str] = []
        for indicator in defn.indicators:
            if _found(indicator):
                matched.append(indicator)

        if not defn.indicators:
            base_confidence = 0.0
        else:
            base_confidence = len(matched) / len(defn.indicators)

        # Бонус за совпадение шаблона
        if self._matcher is not None and defn.templates:
            for tmpl in defn.templates:
                try:
                    if self._matcher.find_one(screenshot, tmpl, threshold=0.75) is not None:
                        base_confidence = min(1.0, base_confidence + 0.3)
                        break
                except Exception:
                    pass

        return base_confidence, matched
