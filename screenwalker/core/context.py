"""RunContext — изменяемое состояние выполнения одного сценария.

Контекст передаётся через каждый шаг и вызов компьютерного зрения, чтобы все
компоненты разделяли одни переменные, историю и скриншоты без
жёсткой связи друг с другом.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class StepRecord:
    """Неизменяемая запись о выполненном шаге.

    Attributes:
        step_id: Идентификатор шага.
        action: Строка действия (например, "click").
        success: Завершился ли шаг без ошибок.
        elapsed: Реальное время выполнения в секундах.
        screenshot_path: Путь к скриншоту после шага (если есть).
        error: Сообщение об ошибке при неуспешном шаге.
        metadata: Произвольные пары ключ-значение для отладки.
    """

    step_id: str
    action: str
    success: bool
    elapsed: float
    screenshot_path: Path | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RunContext:
    """Хранит всё изменяемое состояние одного запуска сценария.

    Контекст намеренно не является frozen dataclass — шаги должны
    обновлять переменные, добавлять записи истории и кэшировать скриншоты
    в процессе выполнения.

    Attributes:
        scenario_name: Читаемое имя сценария из YAML.
        run_id: Уникальный идентификатор запуска (на основе временной метки).
        variables: Изменяемый набор переменных, подставляемых в текст и запросы шагов.
        history: Упорядоченный список записей о выполненных шагах.
        current_screen: Идентификатор последнего обнаруженного состояния экрана.
        last_screenshot: Последний захваченный PIL-образ.
        output_dir: Директория для записи логов и скриншотов.
        dry_run: Если True, действия логируются, но не выполняются.
    """

    def __init__(
        self,
        scenario_name: str,
        variables: dict[str, str] | None = None,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        """Инициализация RunContext.

        Args:
            scenario_name: Имя выполняемого сценария.
            variables: Начальный словарь переменных (из YAML и переопределений CLI).
            output_dir: Корневая директория для логов и скриншотов.
            dry_run: Пропускать ли фактическое выполнение действий.
        """
        self.scenario_name = scenario_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.variables: dict[str, str] = variables or {}
        self.history: list[StepRecord] = []
        self.current_screen: str | None = None
        self.last_screenshot: Image.Image | None = None
        self.output_dir: Path = output_dir or Path("logs") / self.run_id
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Подстановка переменных
    # ------------------------------------------------------------------

    def interpolate(self, text: str) -> str:
        """Заменяет плейсхолдеры ``{{ variable_name }}`` в строке *text*.

        Args:
            text: Шаблонная строка, содержащая токены ``{{ var }}``.

        Returns:
            Строка с подставленными известными переменными. Неизвестные переменные
            остаются без изменений.

        Example:
            >>> ctx = RunContext("demo", variables={"user": "admin"})
            >>> ctx.interpolate("Hello {{ user }}!")
            'Hello admin!'
        """
        # TODO: реализовать подстановку в стиле Jinja2 через self.variables
        def replace(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            return self.variables.get(key, match.group(0))

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, text)

    def set_variable(self, key: str, value: str) -> None:
        """Устанавливает или обновляет переменную времени выполнения.

        Args:
            key: Имя переменной.
            value: Новое значение (всегда сохраняется как строка).
        """
        self.variables[key] = value

    # ------------------------------------------------------------------
    # Управление историей
    # ------------------------------------------------------------------

    def record_step(self, record: StepRecord) -> None:
        """Добавляет запись о выполненном шаге в историю.

        Args:
            record: Экземпляр :class:`StepRecord` для добавления.
        """
        self.history.append(record)

    @property
    def failed_steps(self) -> list[StepRecord]:
        """Возвращает все записи шагов, завершившихся с ошибкой."""
        return [r for r in self.history if not r.success]

    @property
    def step_count(self) -> int:
        """Общее число выполненных шагов (успешных и неуспешных)."""
        return len(self.history)

    # ------------------------------------------------------------------
    # Вспомогательные методы для скриншотов
    # ------------------------------------------------------------------

    def save_screenshot(self, image: Image.Image, label: str = "step") -> Path:
        """Сохраняет PIL-образ в директорию вывода текущего запуска.

        Args:
            image: PIL-образ для сохранения.
            label: Метка для имени файла (идентификатор шага или описание).

        Returns:
            Абсолютный путь к сохранённому файлу.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe = label.replace(" ", "_").replace("/", "-").replace("\\", "-")
        path = self.output_dir / f"{safe}.png"
        image.save(path, "PNG")
        return path.resolve()

    def update_screenshot(self, image: Image.Image) -> None:
        """Обновляет кэшированный последний скриншот.

        Args:
            image: Последний захваченный образ экрана.
        """
        self.last_screenshot = image

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RunContext(scenario={self.scenario_name!r}, "
            f"run_id={self.run_id!r}, "
            f"steps={self.step_count}, "
            f"dry_run={self.dry_run})"
        )
