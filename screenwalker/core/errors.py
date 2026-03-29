"""Пользовательские исключения среды выполнения ScreenWalker.

Все исключения наследуются от :class:`ScenarioError`, что позволяет
перехватывать всю иерархию одной конструкцией ``except ScenarioError``,
сохраняя возможность обработки конкретных сбоев.
"""

from __future__ import annotations


class ScenarioError(Exception):
    """Базовый класс для всех ошибок выполнения ScreenWalker."""

    def __init__(self, message: str, step_id: str | None = None) -> None:
        """Инициализация ScenarioError.

        Args:
            message: Описание ошибки в виде строки.
            step_id: Идентификатор шага, вызвавшего ошибку (необязательно).
        """
        super().__init__(message)
        self.step_id = step_id

    def __str__(self) -> str:
        if self.step_id:
            return f"[step={self.step_id}] {super().__str__()}"
        return super().__str__()


class StepTimeout(ScenarioError):
    """Вызывается, когда элемент не найден в течение заданного таймаута.

    Attributes:
        step_id: Идентификатор шага, превысившего таймаут.
        timeout: Настроенный таймаут в секундах.
        query: Запрос поиска элемента, который не был выполнен.
    """

    def __init__(self, step_id: str, timeout: float, query: str) -> None:
        """Инициализация StepTimeout.

        Args:
            step_id: Идентификатор шага, превысившего таймаут.
            timeout: Продолжительность таймаута в секундах.
            query: Поисковый запрос (текст, имя шаблона и т.д.).
        """
        super().__init__(
            f"Element not found within {timeout}s: {query!r}",
            step_id=step_id,
        )
        self.timeout = timeout
        self.query = query


class ElementNotFound(ScenarioError):
    """Вызывается, когда элемент не найден после перебора всех методов поиска.

    Attributes:
        step_id: Идентификатор шага, в котором элемент не найден.
        query: Поисковый запрос элемента.
        methods: Методы компьютерного зрения, которые были применены.
    """

    def __init__(self, step_id: str, query: str, methods: list[str]) -> None:
        """Инициализация ElementNotFound.

        Args:
            step_id: Идентификатор упавшего шага.
            query: Строка поискового запроса или имя шаблона.
            methods: Список использованных методов поиска (например, ['ocr', 'template']).
        """
        methods_str = ", ".join(methods)
        super().__init__(
            f"Element {query!r} not found. Methods tried: [{methods_str}]",
            step_id=step_id,
        )
        self.query = query
        self.methods = methods


class ScreenMismatch(ScenarioError):
    """Вызывается, когда текущее состояние экрана не совпадает с ожидаемым.

    Attributes:
        step_id: Идентификатор шага, обнаружившего несоответствие.
        expected: Ожидаемый идентификатор экрана.
        actual: Фактически обнаруженный идентификатор экрана (или None, если неизвестен).
    """

    def __init__(self, step_id: str, expected: str, actual: str | None) -> None:
        """Инициализация ScreenMismatch.

        Args:
            step_id: Идентификатор упавшего шага.
            expected: Ожидаемый идентификатор состояния экрана.
            actual: Обнаруженный идентификатор состояния экрана, или None если не распознан.
        """
        actual_desc = actual if actual else "unknown"
        super().__init__(
            f"Screen mismatch: expected {expected!r}, got {actual_desc!r}",
            step_id=step_id,
        )
        self.expected = expected
        self.actual = actual


class ScenarioValidationError(ScenarioError):
    """Вызывается при ошибке валидации схемы YAML-сценария до начала выполнения."""


class ActionError(ScenarioError):
    """Вызывается при сбое низкоуровневого действия (клик мышью, нажатие клавиши и т.д.)."""


class ConfigurationError(ScenarioError):
    """Вызывается при отсутствии или некорректности обязательной конфигурации."""
