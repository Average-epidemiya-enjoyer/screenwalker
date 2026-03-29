"""Логгер шагов — записывает выполнение каждого шага с JSONL-метаданными и скриншотами.

Каждый запуск создаёт директорию вида::

    logs/2024-01-15_14-30-00_create_ticket/
        steps.jsonl           ← один JSON-объект на строку
        screenshots/
            step_001_before.png
            step_001_after.png
            step_002_before.png
            ...

Предоставляются два взаимодополняющих API:

* :meth:`StepLogger.log_step` — обратно совместимый; принимает
  :class:`~screenwalker.core.context.StepRecord` движка и извлекает
  метаданные vision из словаря ``metadata``.
* :meth:`StepLogger.log_result` — расширенный API, принимающий полностью
  заполненный :class:`StepResult` с явными полями vision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

from screenwalker.core.context import StepRecord

_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# StepResult — расширенная запись шага для JSONL-сериализации
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Расширенная запись выполненного шага для JSONL-логирования.

    Attributes:
        step_id: Идентификатор шага.
        scenario: Название родительского сценария.
        action: Выполненное действие (например, ``"click"``, ``"type"``).
        success: Завершился ли шаг без ошибки.
        duration_ms: Фактическое время выполнения в миллисекундах.
        timestamp: ISO-8601 UTC временная метка выполнения.
        find_target: Текст / шаблон / метка, по которым выполнялся поиск.
        method_used: Метод vision, нашедший элемент
            (``ocr`` / ``template`` / ``yolo`` / ``cache``).
        confidence: Уверенность совпадения в диапазоне ``[0.0, 1.0]``.
        bbox: Ограничивающий прямоугольник ``(x, y, w, h)`` найденного элемента.
        found_text: Фактический текст, возвращённый OCR (может отличаться от
            *find_target* при активном нечётком сопоставлении).
        error: Сообщение об ошибке, если шаг завершился неудачей.
    """

    step_id: str
    scenario: str
    action: str
    success: bool
    duration_ms: float
    timestamp: str
    find_target: str | None = None
    method_used: str | None = None
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    found_text: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# StepLogger
# ---------------------------------------------------------------------------


class StepLogger:
    """Записывает результаты выполнения шагов в JSONL-файл со скриншотами.

    Attributes:
        output_dir: Корневая директория текущего запуска (создаётся лениво).
        save_screenshots: Сохранять скриншоты до/после для каждого шага.
        save_on_failure: Всегда сохранять скриншот при неудаче шага, даже если
            *save_screenshots* равно ``False``.
    """

    def __init__(
        self,
        output_dir: Path | str,
        save_screenshots: bool = True,
        save_on_failure: bool = True,
    ) -> None:
        """Инициализировать StepLogger.

        Args:
            output_dir: Директория вывода для текущего запуска (например,
                ``logs/2024-01-15_14-30-00_create_ticket``).
            save_screenshots: Сохранять PNG-скриншоты для каждого шага.
            save_on_failure: Переопределить *save_screenshots* для неудавшихся шагов.
        """
        self.output_dir = Path(output_dir)
        self.save_screenshots = save_screenshots
        self.save_on_failure = save_on_failure
        self._step_index: int = 0
        self._scenario_name: str = ""

    # ------------------------------------------------------------------
    # Производные пути
    # ------------------------------------------------------------------

    @property
    def jsonl_path(self) -> Path:
        """Путь к JSONL-файлу логов текущего запуска."""
        return self.output_dir / "steps.jsonl"

    @property
    def screenshots_dir(self) -> Path:
        """Директория, в которой хранятся скриншоты шагов."""
        return self.output_dir / "screenshots"

    # ------------------------------------------------------------------
    # Основной API — StepResult
    # ------------------------------------------------------------------

    def log_result(
        self,
        result: StepResult,
        screenshot_before: Image.Image | None = None,
        screenshot_after: Image.Image | None = None,
    ) -> None:
        """Записать JSONL-запись и опционально сохранить скриншоты до/после.

        Args:
            result: Расширенная запись шага для сохранения.
            screenshot_before: Скриншот, снятый непосредственно перед действием.
            screenshot_after: Скриншот, снятый непосредственно после действия.
        """
        self._step_index += 1
        idx = self._step_index

        # -- Скриншоты --------------------------------------------------
        if screenshot_before is not None and self.save_screenshots:
            self._save_screenshot(screenshot_before, f"step_{idx:03d}_before.png")

        should_save_after = self.save_screenshots or (
            not result.success and self.save_on_failure
        )
        if screenshot_after is not None and should_save_after:
            self._save_screenshot(screenshot_after, f"step_{idx:03d}_after.png")

        # -- JSONL --------------------------------------------------------
        self._append_jsonl(
            {
                "timestamp": result.timestamp,
                "step_id": result.step_id,
                "scenario": result.scenario,
                "action": result.action,
                "find_target": result.find_target,
                "method_used": result.method_used,
                "confidence": result.confidence,
                "success": result.success,
                "duration_ms": round(result.duration_ms, 2),
                "error": result.error,
                "bbox": list(result.bbox) if result.bbox else None,
                "found_text": result.found_text,
            }
        )

        # -- structlog ----------------------------------------------------
        level = "info" if result.success else "warning"
        getattr(_log, level)(
            "step_executed",
            step_id=result.step_id,
            action=result.action,
            success=result.success,
            duration_ms=round(result.duration_ms, 1),
            method=result.method_used,
            error=result.error,
        )

    # ------------------------------------------------------------------
    # Обратно совместимый API — StepRecord
    # ------------------------------------------------------------------

    def log_step(
        self,
        record: StepRecord,
        screenshot: Image.Image | None = None,
    ) -> None:
        """Записать шаг из :class:`~screenwalker.core.context.StepRecord`.

        Извлекает метаданные vision (``find_target``, ``method_used``,
        ``confidence``, ``bbox``, ``found_text``) из
        ``record.metadata`` и делегирует вызов :meth:`log_result`.

        Args:
            record: Запись выполнения шага, созданная движком.
            screenshot: Опциональный скриншот, снятый после шага.
        """
        meta: dict[str, Any] = record.metadata or {}

        bbox: tuple[int, int, int, int] | None = None
        bbox_raw = meta.get("bbox")
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = (
                int(bbox_raw[0]),
                int(bbox_raw[1]),
                int(bbox_raw[2]),
                int(bbox_raw[3]),
            )

        result = StepResult(
            step_id=record.step_id,
            scenario=self._scenario_name,
            action=record.action,
            success=record.success,
            duration_ms=record.elapsed * 1000.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            find_target=meta.get("find_target"),
            method_used=meta.get("method_used"),
            confidence=meta.get("confidence"),
            bbox=bbox,
            found_text=meta.get("found_text"),
            error=record.error,
        )
        self.log_result(result, screenshot_after=screenshot)

    # ------------------------------------------------------------------
    # События уровня сценария
    # ------------------------------------------------------------------

    def log_scenario_start(self, scenario_name: str, variable_count: int) -> None:
        """Записать начало выполнения сценария.

        Args:
            scenario_name: Отображаемое имя сценария.
            variable_count: Количество загруженных переменных времени выполнения.
        """
        self._scenario_name = scenario_name
        self._append_jsonl(
            {
                "type": "scenario_start",
                "scenario": scenario_name,
                "variables": variable_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        _log.info("scenario_started", scenario=scenario_name, variables=variable_count)

    def log_scenario_end(
        self,
        scenario_name: str,
        total_steps: int,
        failed_steps: int,
        elapsed: float,
    ) -> None:
        """Записать завершение выполнения сценария.

        Args:
            scenario_name: Отображаемое имя сценария.
            total_steps: Общее количество выполненных шагов.
            failed_steps: Шаги, завершившиеся с ошибкой.
            elapsed: Общее фактическое время выполнения в секундах.
        """
        success = failed_steps == 0
        self._append_jsonl(
            {
                "type": "scenario_end",
                "scenario": scenario_name,
                "total_steps": total_steps,
                "failed_steps": failed_steps,
                "elapsed_s": round(elapsed, 3),
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        level = "info" if success else "error"
        getattr(_log, level)(
            "scenario_ended",
            scenario=scenario_name,
            total_steps=total_steps,
            failed_steps=failed_steps,
            elapsed=round(elapsed, 2),
            success=success,
        )

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _save_screenshot(self, image: Image.Image, filename: str) -> Path:
        """Сохранить *image* в поддиректорию скриншотов.

        Args:
            image: PIL-изображение для сохранения.
            filename: Имя файла назначения (без компонента пути).

        Returns:
            Абсолютный путь к сохранённому файлу.
        """
        dest = self.screenshots_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, "PNG")
        return dest.resolve()

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        """Добавить *entry* как одну JSON-строку в JSONL-файл запуска.

        Args:
            entry: Словарь для сериализации.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_step_screenshot(
        self, image: Image.Image, step_id: str, success: bool
    ) -> Path:
        """Устаревший вспомогательный метод для вызовов, передающих step_id + success отдельно.

        Args:
            image: PIL-скриншот.
            step_id: Идентификатор шага, используемый в имени файла.
            success: Успешно ли выполнился шаг (влияет на префикс имени файла).

        Returns:
            Путь к сохранённому файлу.
        """
        prefix = "ok" if success else "fail"
        slug = step_id.replace(" ", "_").replace("/", "-")
        filename = f"{self._step_index:03d}_{prefix}_{slug}.png"
        return self._save_screenshot(image, filename)
