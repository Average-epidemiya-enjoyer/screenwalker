"""ScenarioEngine — главный оркестратор, реализующий конечный автомат выполнения шагов.

Движок загружает сценарий из YAML, валидирует его и прогоняет каждый шаг
через конвейер vision → action → assert, управляя повторами,
фолбэками и :class:`~screenwalker.core.context.RunContext`.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml
from PIL import Image

from screenwalker.core.context import RunContext, StepRecord
from screenwalker.core.errors import (
    ActionError,
    ElementNotFound,
    ScenarioError,
    ScenarioValidationError,
    ScreenMismatch,
    StepTimeout,
)
from screenwalker.core.step import FindMethod, FindSpec, OnFailure, RecoveryTrigger, Step, StepAction
from screenwalker.learning.cache import ActionCache
from screenwalker.learning.logger import StepLogger
from screenwalker.utils.config import AppConfig, merge_scenario_config
from screenwalker.vision.screen_state import BBox, FindResult

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class _RecoveryGoto(Exception):
    """Внутренний сигнал: перейти к именованному шагу после recovery-действия."""

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        super().__init__(f"goto:{step_id}")


def _build_region(raw: str | list[int] | None) -> BBox | None:
    """Преобразует «сырую» спецификацию региона в BBox.

    Args:
        raw: Либо ``None``, либо строка с именем региона (игнорируется — возвращает None),
            либо список ``[x, y, w, h]``.

    Returns:
        BBox, если *raw* — список; иначе None.
    """
    if raw is None or isinstance(raw, str):
        return None
    if isinstance(raw, list) and len(raw) == 4:
        return BBox(x=raw[0], y=raw[1], w=raw[2], h=raw[3])
    return None


class ScenarioEngine:
    """Выполняет сценарий YAML как упорядоченный конечный автомат.

    Каждый шаг обрабатывается последовательно:

    1. **Interpolate** — подставляет ``{{ переменные }}`` в запросы и тексты.
    2. **Capture** — делает свежий скриншот.
    3. **Verify screen** — проверяет :attr:`~screenwalker.core.step.Step.expect_screen`
       относительно текущего состояния UI (если настроен анализатор экрана).
    4. **Locate** — запускает выбранный vision-finder (кэш → OCR / template / YOLO).
    5. **Act** — делегирует вызов соответствующему обработчику действия.
    6. **Record** — записывает :class:`~screenwalker.core.context.StepRecord`.
    7. **Retry** — при ошибке повторяет попытку до ``step.retries`` раз с
       экспоненциальной задержкой.

    Attributes:
        scenario_name: Человекочитаемое имя из YAML.
        steps: Упорядоченный список объектов :class:`~screenwalker.core.step.Step`.
        teardown_steps: Шаги, выполняемые после основных шагов (в том числе при ошибке).
        context: Изменяемый :class:`~screenwalker.core.context.RunContext`.
        config: Валидированный :class:`~screenwalker.utils.config.AppConfig`.
    """

    def __init__(
        self,
        scenario_name: str,
        steps: list[Step],
        teardown_steps: list[Step],
        context: RunContext,
        config: AppConfig,
        *,
        capture: Any = None,
        ocr_engine: Any = None,
        template_matcher: Any = None,
        screen_analyzer: Any = None,
        mouse: Any = None,
        keyboard: Any = None,
        clipboard: Any = None,
        cache: ActionCache | None = None,
        step_logger: StepLogger | None = None,
        detector: Any = None,
        popup_handler: Any = None,
    ) -> None:
        """Инициализирует ScenarioEngine со всеми подсистемами.

        Args:
            scenario_name: Отображаемое имя сценария.
            steps: Основные шаги выполнения.
            teardown_steps: Шаги, запускаемые после основных вне зависимости от результата.
            context: Контекст запуска, общий для всех шагов.
            config: Валидированная конфигурация приложения.
            capture: Переопределяет :class:`~screenwalker.vision.capture.ScreenCapture`.
            ocr_engine: Переопределяет OCR-движок (TesseractEngine или совместимый).
            template_matcher: Переопределяет :class:`~screenwalker.vision.template_match.TemplateMatcher`.
            screen_analyzer: Переопределяет :class:`~screenwalker.vision.screen_state.ScreenStateAnalyzer`.
            mouse: Переопределяет :class:`~screenwalker.actions.mouse.MouseController`.
            keyboard: Переопределяет :class:`~screenwalker.actions.keyboard.KeyboardController`.
            clipboard: Переопределяет :class:`~screenwalker.actions.clipboard.ClipboardManager`.
            cache: Переопределяет :class:`~screenwalker.learning.cache.ActionCache`.
            step_logger: Переопределяет :class:`~screenwalker.learning.logger.StepLogger`.
            detector: Переопределяет YOLO-детектор (UIElementDetector или совместимый).
        """
        self.scenario_name = scenario_name
        self.steps = steps
        self.teardown_steps = teardown_steps
        self.context = context
        self.config = config
        self._log = structlog.get_logger(__name__).bind(scenario=scenario_name)

        # ── Vision-подсистемы ────────────────────────────────────────────────
        self._capture = capture or self._build_capture(config)
        self._ocr = ocr_engine  # ленивая инициализация: None; вызывающий код подставляет для реальных запусков
        self._matcher = template_matcher
        self._screen_analyzer = screen_analyzer
        self._detector = detector  # YOLO-детектор; None → YOLO-фолбэк отключён
        self._popup_handler = popup_handler

        # ── Action-подсистемы ────────────────────────────────────────────────
        self._mouse = mouse or self._build_mouse(config)
        self._keyboard = keyboard or self._build_keyboard(config)
        self._clipboard = clipboard or self._build_clipboard(config)

        # ── Learning-подсистемы ──────────────────────────────────────────────
        self._cache = cache or ActionCache(
            path=Path(config.learning.cache_path),
            ttl=config.learning.cache_ttl_seconds,
            enabled=config.learning.cache_enabled,
        )
        self._step_logger = step_logger or StepLogger(
            output_dir=context.output_dir,
            save_screenshots=config.logging.save_screenshots,
            save_on_failure=config.logging.save_on_failure,
        )

    # ------------------------------------------------------------------
    # Построители подсистем
    # ------------------------------------------------------------------

    @staticmethod
    def _build_capture(config: AppConfig) -> Any:
        from screenwalker.vision.capture import ScreenCapture
        return ScreenCapture(screenshot_delay=config.timeouts.screenshot_delay)

    @staticmethod
    def _build_mouse(config: AppConfig) -> Any:
        from screenwalker.actions.mouse import MouseController
        return MouseController(
            move_duration=config.actions.mouse_move_duration,
            humanize=config.actions.humanize,
            humanize_offset_px=config.actions.humanize_offset_px,
            humanize_delay_min_ms=config.actions.humanize_delay_min_ms,
            humanize_delay_max_ms=config.actions.humanize_delay_max_ms,
        )

    @staticmethod
    def _build_keyboard(config: AppConfig) -> Any:
        from screenwalker.actions.keyboard import KeyboardController
        return KeyboardController(typing_interval=config.actions.typing_interval)

    @staticmethod
    def _build_clipboard(config: AppConfig) -> Any:
        from screenwalker.actions.clipboard import ClipboardManager
        return ClipboardManager(settle_delay=config.actions.clipboard_settle_delay)

    # ------------------------------------------------------------------
    # Фабрика
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        config: AppConfig | None = None,
        variable_overrides: dict[str, str] | None = None,
    ) -> "ScenarioEngine":
        """Загружает и валидирует YAML-файл сценария, возвращая готовый движок.

        Args:
            path: Путь к YAML-файлу сценария.
            config: Предзагруженная конфигурация приложения; если None — загружаются значения по умолчанию.
            variable_overrides: Переопределения из CLI-опции ``--var``, накладываемые поверх
                секции ``variables`` сценария.

        Returns:
            Полностью инициализированный :class:`ScenarioEngine`.

        Raises:
            ScenarioValidationError: Если YAML невалиден или отсутствуют обязательные поля.
            FileNotFoundError: Если *path* не существует.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ScenarioValidationError("Scenario YAML must be a mapping at top level")

        scenario_name: str = raw.get("name", path.stem)
        raw_variables: dict[str, str] = raw.get("variables", {})
        variables = {**raw_variables, **(variable_overrides or {})}

        from screenwalker.utils.config import load_config

        base_config = config or load_config(None)

        # Применяем переопределения конфига из секции сценария
        scenario_config_raw = raw.get("config", {})
        resolved_config = (
            merge_scenario_config(base_config, scenario_config_raw)
            if scenario_config_raw
            else base_config
        )

        steps = cls._parse_steps(raw.get("steps", []))
        teardown_steps = cls._parse_steps(raw.get("teardown", []))

        context = RunContext(
            scenario_name=scenario_name,
            variables=variables,
            output_dir=Path(resolved_config.logging.output_dir) / scenario_name,
        )

        engine = cls(
            scenario_name=scenario_name,
            steps=steps,
            teardown_steps=teardown_steps,
            context=context,
            config=resolved_config,
        )

        # Загружаем OCR-движок, если настроен
        if resolved_config.vision.ocr_engine == "tesseract":
            try:
                from screenwalker.vision.ocr import TesseractEngine
                engine._ocr = TesseractEngine(
                    lang=resolved_config.vision.ocr_lang,
                    config=resolved_config.vision.ocr_config,
                    preprocess=resolved_config.vision.ocr_preprocess,
                )
            except ImportError:
                logger.warning("Tesseract not available — OCR steps will fail")

        # Загружаем YOLO-детектор, если включён
        if getattr(resolved_config.vision, "yolo_enabled", False):
            try:
                from screenwalker.vision.detector import create_detector
                engine._detector = create_detector(resolved_config.vision)
            except Exception as exc:
                logger.warning("YOLO detector init failed", error=str(exc))

        # Загружаем template matcher, если существует директория с шаблонами
        templates_dir = path.parent / "templates"
        if templates_dir.exists():
            try:
                from screenwalker.vision.template_match import TemplateMatcher
                engine._matcher = TemplateMatcher(templates_dir=templates_dir)
            except ImportError:
                logger.warning("OpenCV not available — template steps will fail")

        # Загружаем popup handler, если существует popups.yaml или конфиг попапов
        popups_config_path = path.parent / "popups.yaml"
        if popups_config_path.exists() and engine._ocr is not None:
            try:
                from screenwalker.vision.popup import PopupHandler
                engine._popup_handler = PopupHandler.from_yaml(
                    popups_config_path,
                    ocr_engine=engine._ocr,
                    mouse=engine._mouse,
                    template_matcher=engine._matcher,
                )
            except Exception as exc:
                logger.warning("Failed to init popup handler", error=str(exc))

        # Загружаем fingerprints экранов из YAML сценария
        screens_config = raw.get("screens", {})
        if screens_config and engine._ocr is not None:
            try:
                from screenwalker.vision.screen_state import ScreenStateAnalyzer, ScreenFingerprint
                fingerprints = []
                for state_id, cfg in screens_config.items():
                    fp = ScreenFingerprint(
                        screen_id=state_id,
                        required_texts=cfg.get("required_texts", []),
                        forbidden_texts=cfg.get("forbidden_texts", []),
                        required_templates=cfg.get("required_templates", []),
                        optional_texts=cfg.get("optional_texts", []),
                        match_threshold=cfg.get("match_threshold", 0.75),
                    )
                    fingerprints.append(fp)
                engine._screen_analyzer = ScreenStateAnalyzer(
                    fingerprints=fingerprints,
                    ocr_engine=engine._ocr,
                    template_matcher=engine._matcher,
                )
            except Exception as exc:
                logger.warning("Failed to init screen analyzer", error=str(exc))

        return engine

    # ------------------------------------------------------------------
    # Выполнение
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Выполняет все шаги сценария, затем запускает teardown-шаги.

        Поддерживает recovery-действие ``goto``, переводящее выполнение к именованному шагу.

        Raises:
            ScenarioError: Перебрасывается после teardown, если любой шаг с
                ``on_failure=abort`` завершился ошибкой.
        """
        run_start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        self._log.info("Scenario started", step_count=len(self.steps))
        self._step_logger.log_scenario_start(
            self.scenario_name, len(self.context.variables)
        )
        failure: ScenarioError | None = None

        try:
            idx = 0
            while idx < len(self.steps):
                step = self.steps[idx]
                try:
                    self._execute_step(step)
                    idx += 1
                except _RecoveryGoto as goto_req:
                    step_ids = [s.id for s in self.steps]
                    if goto_req.step_id in step_ids:
                        idx = step_ids.index(goto_req.step_id)
                        self._log.info("recovery.goto", target=goto_req.step_id)
                    else:
                        self._log.warning(
                            "recovery.goto_target_not_found",
                            target=goto_req.step_id,
                        )
                        idx += 1
                except ScenarioError as exc:
                    if step.on_failure in (OnFailure.ABORT,):
                        failure = exc
                        break
                    self._log.warning(
                        "Step failed, continuing",
                        step_id=step.id,
                        on_failure=step.on_failure.value,
                        error=str(exc),
                    )
                    idx += 1
        finally:
            self._run_teardown()
            elapsed = time.monotonic() - run_start
            self._step_logger.log_scenario_end(
                self.scenario_name,
                self.context.step_count,
                len(self.context.failed_steps),
                elapsed,
            )

        # Автогенерация HTML-отчёта (если включено в конфиге)
        if self.config.report.enabled:
            self._generate_report(started_at, datetime.now(timezone.utc))

        if failure:
            raise failure

        self._log.info(
            "Scenario finished",
            total=self.context.step_count,
            failed=len(self.context.failed_steps),
        )

    def dry_run(self) -> None:
        """Выводит запланированные шаги без выполнения каких-либо действий."""
        self._log.info("DRY RUN — no actions will be executed")
        for i, step in enumerate(self.steps, start=1):
            action_repr = f"[{step.action.value}]"
            find_repr = (
                f"find={step.find.method.value}:{step.find.query!r}"
                if step.find
                else ""
            )
            print(f"  {i:>3}. {step.id:<30} {action_repr:<20} {find_repr}")

    # ------------------------------------------------------------------
    # Выполнение шага (с повторами)
    # ------------------------------------------------------------------

    def _execute_step(self, step: Step) -> None:
        """Выполняет один шаг с повторами до ``step.retries`` раз.

        Recovery-действия применяются между попытками, если триггер ошибки
        соответствует записи в ``step.recovery``.

        Args:
            step: Шаг для выполнения.

        Raises:
            ScenarioError: После исчерпания всех попыток.
            _RecoveryGoto: Если recovery-действие указывает ``then: goto:<id>``.
        """
        log = self._log.bind(step_id=step.id, action=step.action.value)
        max_attempts = max(1, step.retries)
        last_exc: ScenarioError | None = None

        for attempt in range(max_attempts):
            if attempt > 0:
                backoff = min(
                    self.config.retry.backoff_base ** attempt,
                    self.config.retry.backoff_max,
                )
                log.info(
                    "Retrying step",
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    backoff=round(backoff, 2),
                )
                time.sleep(backoff)

            try:
                self._execute_step_once(step)
                return
            except (ElementNotFound, StepTimeout) as exc:
                last_exc = exc
                log.warning("Step attempt failed", attempt=attempt + 1, error=str(exc))
                if step.recovery and not self.context.dry_run:
                    trigger = (
                        RecoveryTrigger.TIMEOUT
                        if isinstance(exc, StepTimeout)
                        else RecoveryTrigger.ELEMENT_NOT_FOUND
                    )
                    if not self._apply_recovery(step, trigger):
                        raise
            except ScreenMismatch as exc:
                last_exc = exc
                log.warning("Step attempt failed", attempt=attempt + 1, error=str(exc))
                if step.recovery and not self.context.dry_run:
                    if not self._apply_recovery(step, RecoveryTrigger.SCREEN_MISMATCH):
                        raise
            except ScenarioError as exc:
                last_exc = exc
                log.warning("Step attempt failed", attempt=attempt + 1, error=str(exc))

        assert last_exc is not None
        raise last_exc

    def _execute_step_once(self, step: Step) -> None:
        """Выполняет одну попытку шага через полный конвейер.

        Args:
            step: Шаг для выполнения.

        Raises:
            StepTimeout: Если элемент не найден в течение таймаута.
            ElementNotFound: Если все поисковые методы исчерпаны.
            ScreenMismatch: Если expect_screen не совпадает с текущим экраном.
            ActionError: Если само действие выбросило исключение.
        """
        log = self._log.bind(step_id=step.id, action=step.action.value)
        log.info("Executing step", description=step.description)

        start = time.monotonic()
        success = False
        error: str | None = None
        screenshot: Image.Image | None = None

        try:
            # 1. Интерполяция переменных
            interpolated = self._interpolate_step(step)

            # 2. Снимок экрана
            if not self.context.dry_run:
                screenshot = self._capture.capture_full()
                self.context.update_screenshot(screenshot)

            # 2b. Обнаруживаем и закрываем попап перед выполнением шага
            if not self.context.dry_run and self._popup_handler is not None and screenshot is not None:
                if self._popup_handler.detect_and_handle(
                    screenshot,
                    capture_fn=self._capture.capture_full,
                ):
                    log.info("popup_dismissed_before_step")
                    screenshot = self._capture.capture_full()
                    self.context.update_screenshot(screenshot)

            # 3. Проверяем состояние экрана
            if interpolated.expect_screen and not self.context.dry_run:
                self._verify_screen_state(interpolated, screenshot)

            # 4. Поиск элемента
            find_result: FindResult | None = None
            if interpolated.find is not None and not self.context.dry_run:
                find_result = self._locate_with_retry(interpolated)

            # 5. Выполняем действие
            self._dispatch_action(interpolated, find_result)

            # 6. Пауза после действия
            if interpolated.wait_after > 0:
                time.sleep(interpolated.wait_after)

            success = True

        except ScenarioError as exc:
            error = str(exc)
            raise
        finally:
            elapsed = time.monotonic() - start
            record = StepRecord(
                step_id=step.id,
                action=step.action.value,
                success=success,
                elapsed=elapsed,
                error=error,
            )
            self.context.record_step(record)
            try:
                self._step_logger.log_step(
                    record=record,
                    screenshot=screenshot or self.context.last_screenshot,
                )
            except Exception as exc:
                log.debug("Step logger error (non-fatal)", error=str(exc))

    # ------------------------------------------------------------------
    # Проверка состояния экрана
    # ------------------------------------------------------------------

    def _verify_screen_state(self, step: Step, screenshot: Image.Image | None) -> None:
        """Проверяет, что текущий экран соответствует step.expect_screen.

        Args:
            step: Шаг с заполненным полем expect_screen.
            screenshot: Изображение текущего экрана.

        Raises:
            ScreenMismatch: Если обнаруженный экран не совпадает с ожидаемым.
        """
        if self._screen_analyzer is None or screenshot is None:
            return
        ident = self._screen_analyzer.identify(screenshot)
        self.context.current_screen = ident.screen_id
        if ident.screen_id != step.expect_screen:
            raise ScreenMismatch(
                step_id=step.id,
                expected=step.expect_screen,
                actual=ident.screen_id,
            )

    # ------------------------------------------------------------------
    # Vision-конвейер — поиск элемента
    # ------------------------------------------------------------------

    def _locate_with_retry(self, step: Step) -> FindResult:
        """Опрашивает экран, пока целевой элемент не найден или не истёк таймаут.

        Приоритет на каждом опросе:
          1. Поиск в кэше (быстрый путь)
          2. Основной finder (``step.find``)
          3. Фолбэк finder (``step.fallback``)

        Args:
            step: Интерполированный шаг, чья спецификация ``find`` описывает цель.

        Returns:
            Объект :class:`~screenwalker.vision.screen_state.FindResult`.

        Raises:
            StepTimeout: Если элемент не найден в течение таймаута.
        """
        assert step.find is not None
        timeout = step.timeout or self.config.timeouts.step_default
        poll_interval = self.config.timeouts.poll_interval
        deadline = time.monotonic() + timeout
        log = self._log.bind(step_id=step.id, query=step.find.query)

        first_poll = True
        while True:
            if not first_poll:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_interval, remaining))
            first_poll = False

            # Делаем свежий снимок экрана на каждом опросе
            try:
                image = self._capture.capture_full()
                self.context.update_screenshot(image)
            except Exception as exc:
                log.warning("Screen capture failed during locate", error=str(exc))
                continue

            # 1. Поиск в кэше
            screen_id = self.context.current_screen or ""
            cached = self._cache.lookup(screen_id, step.find.query)
            if cached is not None:
                log.debug("Element found in cache")
                return cached

            # 2. Основной finder
            result = self._find_by_spec(image, step.find, step.id)

            # 3. Фолбэк finder
            if result is None and step.fallback is not None:
                result = self._find_by_spec(image, step.fallback, step.id)

            # 4. YOLO-фолбэк — пробуем детектор, если OCR и template не нашли элемент
            if result is None and self._detector is not None and getattr(self._detector, "available", False):
                if step.find.method != FindMethod.YOLO:
                    yolo_region = _build_region(step.find.region)
                    result = self._detector.find(image, step.find.query, region=yolo_region)
                    if result is not None:
                        log.debug("Element found via YOLO fallback")

            if result is not None:
                # Сохраняем в кэш для последующих шагов
                if screen_id:
                    try:
                        self._cache.store(screen_id, result)
                    except Exception as exc:
                        log.debug("Cache store failed (non-fatal)", error=str(exc))
                return result

            if time.monotonic() >= deadline:
                break

            log.debug("Element not found, retrying...", remaining=round(deadline - time.monotonic(), 1))

        raise StepTimeout(step_id=step.id, timeout=timeout, query=step.find.query)

    def _find_by_spec(
        self, image: Image.Image, spec: FindSpec, step_id: str
    ) -> FindResult | None:
        """Выполняет один проход vision-поиска.

        Args:
            image: Текущий скриншот.
            spec: FindSpec, описывающий метод, запрос, регион и пороги.
            step_id: Используется в warning-логах.

        Returns:
            FindResult, если элемент найден; иначе None.
        """
        region = _build_region(spec.region)

        if spec.method == FindMethod.TEMPLATE:
            if self._matcher is None:
                self._log.warning(
                    "Template matcher not configured — cannot use template find",
                    step_id=step_id,
                )
                return None
            template_name = spec.template or spec.query
            try:
                match = self._matcher.find_one(
                    image, template_name, threshold=spec.threshold, region=region
                )
            except FileNotFoundError as exc:
                self._log.warning("Template not found", step_id=step_id, error=str(exc))
                return None
            if match is None:
                return None
            return FindResult(
                element=match.template_name,
                confidence=match.confidence,
                bbox=match.bbox,
                method="template",
            )

        if spec.method == FindMethod.OCR:
            if self._ocr is None:
                self._log.warning(
                    "OCR engine not configured — cannot use ocr find",
                    step_id=step_id,
                )
                return None
            return self._ocr.find_text(
                image,
                spec.query,
                threshold=spec.threshold,
                region=region,
                fuzzy=spec.fuzzy,
            )

        if spec.method == FindMethod.YOLO:
            if self._detector is None or not getattr(self._detector, "available", False):
                self._log.warning(
                    "YOLO detector not configured or unavailable", step_id=step_id
                )
                return None
            return self._detector.find(
                image,
                spec.query,
                threshold=spec.threshold,
                region=region,
            )

        return None

    # ------------------------------------------------------------------
    # Интерполяция переменных
    # ------------------------------------------------------------------

    def _interpolate_step(self, step: Step) -> Step:
        """Возвращает копию *step* с заменёнными токенами ``{{ var }}``.

        Args:
            step: Исходный шаг (не мутируется).

        Returns:
            Новый шаг со строковыми полями после подстановки переменных.
        """
        ctx = self.context
        updates: dict[str, Any] = {}

        if step.text is not None:
            updates["text"] = ctx.interpolate(step.text)
        if step.target is not None:
            updates["target"] = ctx.interpolate(step.target)
        if step.label is not None:
            updates["label"] = ctx.interpolate(step.label)

        # Интерполируем запрос поиска
        if step.find is not None and step.find.query:
            new_query = ctx.interpolate(step.find.query)
            if new_query != step.find.query:
                updates["find"] = step.find.model_copy(update={"query": new_query})

        if not updates:
            return step
        return step.model_copy(update=updates)

    # ------------------------------------------------------------------
    # Диспетчеризация действий
    # ------------------------------------------------------------------

    def _dispatch_action(self, step: Step, find_result: FindResult | None) -> None:
        """Направляет шаг к соответствующему обработчику действия.

        Args:
            step: Интерполированный шаг для выполнения.
            find_result: Найденный элемент, или None для действий, которым он не нужен.
        """
        if self.context.dry_run:
            logger.debug("DRY RUN — action skipped", action=step.action.value, step_id=step.id)
            return

        action_map = {
            StepAction.CLICK: self._action_click,
            StepAction.DOUBLE_CLICK: self._action_double_click,
            StepAction.RIGHT_CLICK: self._action_right_click,
            StepAction.TYPE: self._action_type,
            StepAction.HOTKEY: self._action_hotkey,
            StepAction.SCROLL: self._action_scroll,
            StepAction.DRAG: self._action_drag,
            StepAction.COPY: self._action_copy,
            StepAction.PASTE: self._action_paste,
            StepAction.ASSERT_VISIBLE: self._action_assert_visible,
            StepAction.ASSERT_TEXT: self._action_assert_text,
            StepAction.WAIT: self._action_wait,
            StepAction.LAUNCH: self._action_launch,
            StepAction.SCREENSHOT: self._action_screenshot,
        }

        handler = action_map.get(step.action)
        if handler is None:
            raise NotImplementedError(f"No handler for action: {step.action}")
        handler(step, find_result)

    # ------------------------------------------------------------------
    # Обработчики действий
    # ------------------------------------------------------------------

    def _resolve_center(
        self, step: Step, find_result: FindResult | None
    ) -> tuple[int, int]:
        """Возвращает экранные координаты (x, y) для выполнения действия.

        Применяет смещение из find spec, если оно задано.

        Args:
            step: Текущий шаг (источник смещения).
            find_result: Найденный элемент.

        Returns:
            (cx, cy) в пикселях экрана.

        Raises:
            ElementNotFound: Если find_result равен None.
        """
        if find_result is None:
            query = step.find.query if step.find else "<no find spec>"
            raise ElementNotFound(step.id, query, ["any"])
        cx, cy = find_result.center
        if step.find and step.find.offset:
            cx += step.find.offset[0]
            cy += step.find.offset[1]
        return cx, cy

    def _action_click(self, step: Step, find_result: FindResult | None) -> None:
        """Левый клик по найденному элементу."""
        cx, cy = self._resolve_center(step, find_result)
        self._mouse.click(cx, cy, button="left")

    def _action_double_click(self, step: Step, find_result: FindResult | None) -> None:
        """Двойной клик по найденному элементу."""
        cx, cy = self._resolve_center(step, find_result)
        self._mouse.double_click(cx, cy)

    def _action_right_click(self, step: Step, find_result: FindResult | None) -> None:
        """Правый клик по найденному элементу."""
        cx, cy = self._resolve_center(step, find_result)
        self._mouse.right_click(cx, cy)

    def _action_type(self, step: Step, find_result: FindResult | None) -> None:
        """Кликает по элементу (если найден), при необходимости очищает его, затем вводит текст."""
        if find_result is not None:
            cx, cy = self._resolve_center(step, find_result)
            self._mouse.click(cx, cy)

        if step.clear_first:
            import pyautogui
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")

        text = step.text or ""
        self._keyboard.type_text(text)

    def _action_hotkey(self, step: Step, find_result: FindResult | None) -> None:
        """Отправляет комбинацию клавиш."""
        if not step.keys:
            raise ActionError("hotkey action requires 'keys'", step_id=step.id)
        self._keyboard.hotkey(*step.keys)

    def _action_scroll(self, step: Step, find_result: FindResult | None) -> None:
        """Прокручивает в позиции элемента или в текущей позиции курсора."""
        direction = str(step.extra.get("direction", "down"))
        clicks = int(step.extra.get("clicks", 3))
        if find_result is not None:
            cx, cy = find_result.center
        else:
            import pyautogui
            pos = pyautogui.position()
            cx, cy = pos.x, pos.y
        self._mouse.scroll(cx, cy, clicks=clicks, direction=direction)

    def _action_drag(self, step: Step, find_result: FindResult | None) -> None:
        """Перетаскивает элемент к целевой позиции.

        Целевая позиция считывается из ``step.extra["to"]`` в формате [x, y].
        """
        if find_result is None:
            query = step.find.query if step.find else "<no find spec>"
            raise ElementNotFound(step.id, query, ["any"])
        to = step.extra.get("to")
        if not to or len(to) != 2:
            raise ActionError(
                "drag action requires extra.to = [x, y]", step_id=step.id
            )
        sx, sy = find_result.center
        self._mouse.drag(sx, sy, int(to[0]), int(to[1]))

    def _action_copy(self, step: Step, find_result: FindResult | None) -> None:
        """Копирует выделенный текст в найденном элементе в переменную контекста."""
        if find_result is not None:
            cx, cy = self._resolve_center(step, find_result)
            self._mouse.click(cx, cy)
        text = self._clipboard.copy_selected()
        var_name = str(step.extra.get("save_to", "clipboard"))
        self.context.set_variable(var_name, text)

    def _action_paste(self, step: Step, find_result: FindResult | None) -> None:
        """Вставляет содержимое буфера обмена в текущую позицию."""
        if find_result is not None:
            cx, cy = self._resolve_center(step, find_result)
            self._mouse.click(cx, cy)
        self._clipboard.paste()

    def _action_assert_visible(self, step: Step, find_result: FindResult | None) -> None:
        """Проверяет, что целевой элемент виден на экране."""
        if find_result is None:
            query = step.find.query if step.find else "<no find spec>"
            raise ElementNotFound(step.id, query, ["any"])
        self._log.info(
            "Assert visible — OK",
            step_id=step.id,
            element=find_result.element,
            confidence=round(find_result.confidence, 3),
        )

    def _action_assert_text(self, step: Step, find_result: FindResult | None) -> None:
        """Проверяет, что текст найденного OCR-элемента совпадает с ожидаемым значением."""
        if find_result is None:
            query = step.find.query if step.find else "<no find spec>"
            raise ElementNotFound(step.id, query, ["ocr"])

        expected = step.text or (step.find.query if step.find else "")
        fuzzy_threshold = step.find.fuzzy_threshold if step.find else 80

        try:
            from rapidfuzz import fuzz
            score = fuzz.partial_ratio(expected.lower(), find_result.element.lower())
        except ImportError:
            # Фолбэк на точное совпадение
            score = 100 if expected.lower() in find_result.element.lower() else 0

        if score < fuzzy_threshold:
            raise ActionError(
                f"assert_text failed: expected {expected!r}, "
                f"got {find_result.element!r} (score={score}, threshold={fuzzy_threshold})",
                step_id=step.id,
            )
        self._log.info(
            "Assert text — OK",
            step_id=step.id,
            expected=expected,
            found=find_result.element,
            score=score,
        )

    def _action_wait(self, step: Step, find_result: FindResult | None) -> None:
        """Ожидает фиксированное время."""
        duration = step.wait or 0.0
        logger.debug("Waiting", seconds=duration, step_id=step.id)
        time.sleep(duration)

    def _action_launch(self, step: Step, find_result: FindResult | None) -> None:
        """Запускает приложение по пути, URL или команде."""
        if not step.target:
            raise ActionError("launch action requires 'target'", step_id=step.id)

        import platform
        system = platform.system()
        try:
            if system == "Windows":
                import os
                os.startfile(step.target)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", step.target])
            else:
                subprocess.Popen([step.target])
        except Exception as exc:
            raise ActionError(
                f"launch failed for target {step.target!r}: {exc}",
                step_id=step.id,
            ) from exc

    def _action_screenshot(self, step: Step, find_result: FindResult | None) -> None:
        """Делает скриншот, сохраняет его и обновляет контекст запуска."""
        image = self._capture.capture_full()
        self.context.update_screenshot(image)
        label = step.label or step.id
        try:
            path = self.context.save_screenshot(image, label)
            self._log.info("Screenshot saved", step_id=step.id, path=str(path))
        except Exception as exc:
            self._log.warning("Screenshot save failed", step_id=step.id, error=str(exc))

    # ------------------------------------------------------------------
    # Стратегии ожидания
    # ------------------------------------------------------------------

    def wait_for_screen_change(
        self,
        timeout: float,
        poll_interval: float = 0.5,
        mse_threshold: float | None = None,
    ) -> bool:
        """Ожидает значимого изменения экрана (метрика MSE).

        Делает начальный снимок и опрашивает экран, пока среднеквадратичная ошибка
        между базовым снимком и текущим экраном не превысит *mse_threshold*.

        Args:
            timeout: Максимальное время ожидания в секундах.
            poll_interval: Интервал между снимками в секундах.
            mse_threshold: Минимальное значение MSE, при котором экран считается изменённым.
                По умолчанию: ``config.vision.screen_change_mse_threshold``.

        Returns:
            True, если экран изменился в течение *timeout*; иначе False.
        """
        threshold = (
            mse_threshold
            if mse_threshold is not None
            else getattr(self.config.vision, "screen_change_mse_threshold", 100.0)
        )
        try:
            import numpy as np
        except ImportError:
            self._log.warning("wait_for_screen_change requires numpy; falling back to fixed wait")
            time.sleep(min(timeout, 1.0))
            return True

        initial = self._capture.capture_full()
        arr_initial = np.array(initial).astype(float)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0.0, remaining)))
            current = self._capture.capture_full()
            arr_current = np.array(current).astype(float)
            if arr_initial.shape == arr_current.shape:
                mse = float(np.mean((arr_initial - arr_current) ** 2))
                if mse >= threshold:
                    self._log.debug("wait_for_screen_change.detected", mse=round(mse, 2))
                    return True

        self._log.warning("wait_for_screen_change.timeout", timeout=timeout)
        return False

    def wait_for_element(
        self,
        find_spec: "FindSpec",
        timeout: float,
        poll_interval: float = 0.5,
    ) -> "FindResult | None":
        """Опрашивает экран, пока целевой элемент не появится.

        Args:
            find_spec: Vision spec, описывающий искомый элемент.
            timeout: Максимальное время ожидания в секундах.
            poll_interval: Интервал между опросами в секундах.

        Returns:
            :class:`~screenwalker.vision.screen_state.FindResult` при обнаружении,
            или ``None``, если истёк таймаут.
        """
        deadline = time.monotonic() + timeout
        first = True
        while time.monotonic() < deadline:
            if not first:
                remaining = deadline - time.monotonic()
                time.sleep(min(poll_interval, max(0.0, remaining)))
            first = False
            try:
                image = self._capture.capture_full()
            except Exception as exc:
                self._log.warning("wait_for_element.capture_failed", error=str(exc))
                continue
            result = self._find_by_spec(image, find_spec, "wait_for_element")
            if result is not None:
                return result

        self._log.warning("wait_for_element.timeout", query=find_spec.query, timeout=timeout)
        return None

    def wait_for_element_gone(
        self,
        find_spec: "FindSpec",
        timeout: float,
        poll_interval: float = 0.5,
    ) -> bool:
        """Опрашивает экран, пока целевой элемент не исчезнет.

        Args:
            find_spec: Vision spec, описывающий отслеживаемый элемент.
            timeout: Максимальное время ожидания в секундах.
            poll_interval: Интервал между опросами в секундах.

        Returns:
            True, если элемент исчез в течение *timeout*; иначе False.
        """
        deadline = time.monotonic() + timeout
        first = True
        while time.monotonic() < deadline:
            if not first:
                remaining = deadline - time.monotonic()
                time.sleep(min(poll_interval, max(0.0, remaining)))
            first = False
            try:
                image = self._capture.capture_full()
            except Exception as exc:
                self._log.warning("wait_for_element_gone.capture_failed", error=str(exc))
                continue
            result = self._find_by_spec(image, find_spec, "wait_for_element_gone")
            if result is None:
                self._log.debug("wait_for_element_gone.gone", query=find_spec.query)
                return True

        self._log.warning("wait_for_element_gone.timeout", query=find_spec.query, timeout=timeout)
        return False

    # ------------------------------------------------------------------
    # Recovery (восстановление после ошибки)
    # ------------------------------------------------------------------

    def _apply_recovery(self, step: Step, trigger: RecoveryTrigger) -> bool:
        """Выполняет recovery-действие, соответствующее *trigger*.

        Args:
            step: Шаг, который завершился ошибкой.
            trigger: Условие ошибки, которое сработало.

        Returns:
            True — продолжать повторы; False — прервать (вызывающий код должен перебросить исключение).

        Raises:
            _RecoveryGoto: Если recovery указывает ``then: goto:<step_id>``.
        """
        matching = [r for r in step.recovery if r.trigger == trigger]
        if not matching:
            return True

        recovery = matching[0]
        log = self._log.bind(
            step_id=step.id,
            trigger=trigger.value,
            recovery_action=recovery.action,
        )
        log.info("recovery.applying")

        if recovery.action == "screenshot_and_abort":
            if self.context.last_screenshot is not None:
                try:
                    self.context.save_screenshot(
                        self.context.last_screenshot,
                        f"recovery_{step.id}_{trigger.value}",
                    )
                except Exception as exc:
                    log.debug("recovery.screenshot_save_failed", error=str(exc))
            return False

        if recovery.action == "scroll_down":
            try:
                import pyautogui
                pyautogui.scroll(-3)
                time.sleep(0.3)
            except Exception as exc:
                log.warning("recovery.scroll_failed", error=str(exc))

        elif recovery.action == "scroll_up":
            try:
                import pyautogui
                pyautogui.scroll(3)
                time.sleep(0.3)
            except Exception as exc:
                log.warning("recovery.scroll_failed", error=str(exc))

        elif recovery.action == "press_escape":
            try:
                import pyautogui
                pyautogui.press("escape")
                time.sleep(0.3)
            except Exception as exc:
                log.warning("recovery.press_escape_failed", error=str(exc))

        elif recovery.action == "press_key":
            keys = recovery.keys or []
            if keys:
                try:
                    import pyautogui
                    pyautogui.hotkey(*keys)
                    time.sleep(0.3)
                except Exception as exc:
                    log.warning("recovery.press_key_failed", keys=keys, error=str(exc))

        else:
            log.warning("recovery.unknown_action", action=recovery.action)

        if recovery.then and recovery.then.startswith("goto:"):
            target_step_id = recovery.then[5:]
            log.info("recovery.goto_triggered", target=target_step_id)
            raise _RecoveryGoto(target_step_id)

        return True

    # ------------------------------------------------------------------
    # HTML-отчёт
    # ------------------------------------------------------------------

    def _generate_report(self, started_at: datetime, finished_at: datetime) -> None:
        """Сгенерировать HTML-отчёт после завершения сценария.

        Ошибки генерации не прерывают выполнение — только логируются.

        Args:
            started_at: Дата/время начала запуска.
            finished_at: Дата/время завершения запуска.
        """
        try:
            from screenwalker.reporting.html_report import RunReportGenerator, RunResult
            run_result = RunResult.from_context(self.context, started_at, finished_at)
            generator = RunReportGenerator(
                thumbnail_max_width=self.config.report.thumbnail_max_width,
                jpeg_quality=self.config.report.jpeg_quality,
                include_screenshots=self.config.report.include_screenshots,
            )
            output_path = self.context.output_dir / self.config.report.output_path
            report_path = generator.generate(run_result, output_path)
            self._log.info("report.auto_generated", path=str(report_path))
        except Exception as exc:
            self._log.warning("report.generation_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Teardown (завершающие шаги)
    # ------------------------------------------------------------------

    def _run_teardown(self) -> None:
        """Выполняет teardown-шаги, поглощая ошибки, чтобы не скрыть основной сбой."""
        for step in self.teardown_steps:
            try:
                self._execute_step(step)
            except Exception as exc:
                self._log.warning(
                    "Teardown step failed", step_id=step.id, error=str(exc)
                )

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_steps(raw_steps: list[dict[str, Any]]) -> list[Step]:
        """Разбирает список «сырых» словарей в типизированные объекты Step.

        Args:
            raw_steps: Список словарей шагов из YAML.

        Returns:
            Список валидированных экземпляров :class:`~screenwalker.core.step.Step`.

        Raises:
            ScenarioValidationError: Если любой шаг не проходит валидацию Pydantic.
        """
        steps: list[Step] = []
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise ScenarioValidationError(f"Step at index {i} must be a mapping")
            try:
                steps.append(Step(**raw))
            except Exception as exc:
                step_id = raw.get("id", f"<index {i}>")
                raise ScenarioValidationError(
                    f"Validation error in step {step_id!r}: {exc}"
                ) from exc
        return steps
