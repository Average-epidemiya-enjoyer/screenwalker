"""Модель шага — типизированное описание одного шага сценария.

Каждый шаг соответствует одному пользовательскому действию (клик, ввод текста,
проверка и т.д.) плюс опциональной спецификации поиска элемента (как найти
целевой UI-элемент).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RecoveryTrigger(str, Enum):
    """Условие ошибки, активирующее действие восстановления."""

    ELEMENT_NOT_FOUND = "element_not_found"
    SCREEN_MISMATCH = "screen_mismatch"
    TIMEOUT = "timeout"


class RecoveryAction(BaseModel):
    """Одно действие восстановления, выполняемое при сбое шага с определённым триггером.

    Attributes:
        trigger: Условие ошибки, активирующее это восстановление.
        action: Применяемая стратегия — ``scroll_down``, ``scroll_up``,
            ``press_escape``, ``press_key`` или ``screenshot_and_abort``.
        retries: Максимальное число применений этого восстановления (зарезервировано
            для будущего использования; сейчас первое совпадение применяется
            однократно за попытку).
        then: Опциональная директива после восстановления. Поддерживает
            ``"goto:<step_id>"`` для перехода к именованному шагу.
        keys: Имена клавиш для нажатия (для действия ``press_key``).
    """

    trigger: RecoveryTrigger
    action: str
    retries: int = Field(default=1, ge=0)
    then: str | None = None
    keys: list[str] | None = None


class StepAction(str, Enum):
    """Перечисление всех поддерживаемых действий шага."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    COPY = "copy"
    PASTE = "paste"
    ASSERT_VISIBLE = "assert_visible"
    ASSERT_TEXT = "assert_text"
    WAIT = "wait"
    LAUNCH = "launch"
    SCREENSHOT = "screenshot"


class FindMethod(str, Enum):
    """Метод компьютерного зрения для поиска UI-элемента."""

    OCR = "ocr"
    TEMPLATE = "template"
    YOLO = "yolo"


class OnFailure(str, Enum):
    """Поведение при сбое шага."""

    ABORT = "abort"      # Немедленно остановить сценарий (по умолчанию)
    SKIP = "skip"        # Записать ошибку и перейти к следующему шагу
    CONTINUE = "continue"  # Псевдоним для skip; добавлен для читаемости YAML
    RETRY = "retry"      # Повторить шаг до Step.retries раз


class FindSpec(BaseModel):
    """Спецификация поиска UI-элемента перед выполнением действия.

    Attributes:
        method: Метод компьютерного зрения (ocr, template, yolo).
        query: Текст для поиска через OCR или путь к шаблону.
        template: Явный путь к шаблону (альтернатива query для метода template).
        threshold: Минимальный порог уверенности (0.0–1.0).
        region: Именованная область или ограничивающий прямоугольник [x, y, w, h].
        offset: Смещение [dx, dy] в пикселях от центра найденного элемента.
        fuzzy: Использовать ли нечёткое сопоставление текста для OCR.
        fuzzy_threshold: Порог нечёткого сопоставления RapidFuzz (0–100).
    """

    method: FindMethod = FindMethod.OCR
    query: str = ""
    template: str | None = None
    threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    region: str | list[int] | None = None
    offset: list[int] | None = None
    fuzzy: bool = False
    fuzzy_threshold: int = Field(default=80, ge=0, le=100)

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: list[int] | None) -> list[int] | None:
        """Проверяет, что смещение содержит ровно два целых числа (если задано)."""
        if v is not None and len(v) != 2:
            raise ValueError("offset must be a list of exactly 2 integers [dx, dy]")
        return v


class Step(BaseModel):
    """Один шаг сценария.

    Attributes:
        id: Уникальный идентификатор шага внутри сценария.
        description: Читаемое описание, отображаемое в логах.
        action: Выполняемое действие.
        find: Способ поиска целевого элемента (необязательно для ряда действий).
        fallback: Альтернативный поиск при неудаче основного.
        text: Текст для ввода (для действия ``type``).
        keys: Комбинация клавиш (для действия ``hotkey``).
        target: Путь/URL приложения (для действия ``launch``).
        label: Метка скриншота (для действия ``screenshot``).
        wait: Длительность в секундах (для действия ``wait``).
        wait_after: Пауза в секундах после выполнения действия.
        timeout: Переопределяет глобальный таймаут шага для этого шага.
        on_failure: Поведение при сбое шага.
        clear_first: Выделить всё и удалить перед вводом (для действия ``type``).
        extra: Произвольные дополнительные ключи из YAML.
    """

    id: str
    description: str = ""
    action: StepAction
    find: FindSpec | None = None
    fallback: FindSpec | None = None
    text: str | None = None
    keys: list[str] | None = None
    target: str | None = None
    label: str | None = None
    wait: float | None = None
    wait_after: float = 0.0
    timeout: float | None = None
    on_failure: OnFailure = OnFailure.ABORT
    clear_first: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
    expect_screen: str | None = None
    next_step: str | None = None
    retries: int = Field(default=3, ge=0)
    recovery: list[RecoveryAction] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, v: list[str] | None) -> list[str] | None:
        """Проверяет, что имена клавиш являются непустыми строками."""
        if v is not None:
            for key in v:
                if not key.strip():
                    raise ValueError("Key names must be non-empty strings")
        return v
