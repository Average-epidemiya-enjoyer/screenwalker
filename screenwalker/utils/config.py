"""Загрузка конфигурации и валидация через Pydantic.

Конфигурация формируется послойно:
1. Встроенные значения по умолчанию (заданы в Pydantic-моделях).
2. ``config/default.yaml`` — накладывается поверх.
3. YAML уровня проекта, переданный через ``--config``.
4. Блок ``config:`` конкретного сценария в YAML-файле сценария.

Каждый следующий слой перекрывает предыдущие.  Все ключи валидируются
Pydantic, поэтому опечатки в YAML обнаруживаются при старте, а не в процессе выполнения.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic-модели конфигурации
# ---------------------------------------------------------------------------


class TimeoutsConfig(BaseModel):
    """Настройки тайм-аутов.

    Attributes:
        step_default: Тайм-аут для каждого шага по умолчанию (секунды).
        poll_interval: Интервал между повторными попытками поиска элемента.
        screenshot_delay: Пауза перед захватом экрана (ожидание стабилизации UI).
        action_delay: Пауза после каждого действия.
    """

    step_default: float = 15.0
    poll_interval: float = 0.5
    screenshot_delay: float = 0.3
    action_delay: float = 0.1


class VisionConfig(BaseModel):
    """Настройки визуального конвейера.

    Attributes:
        template_threshold: Минимальный порог уверенности при совпадении шаблона.
        multi_scale: Включить многомасштабный поиск шаблона.
        scale_min: Минимальный масштабный коэффициент при многомасштабном поиске.
        scale_max: Максимальный масштабный коэффициент при многомасштабном поиске.
        scale_steps: Количество уровней масштаба.
        ocr_engine: OCR-движок — ``"tesseract"`` или ``"paddleocr"``.
        ocr_lang: Языковой код(ы) для Tesseract.
        ocr_config: Флаги командной строки Tesseract.
        ocr_preprocess: Предварительная обработка изображений перед OCR.
        yolo_enabled: Включить детекцию через YOLO.
        yolo_model_path: Путь к весам модели YOLO.
        yolo_confidence: Минимальный порог уверенности детекции YOLO.
        screen_state_method: Стратегия определения состояния экрана.
        screen_state_threshold: Минимальная уверенность для идентификации состояния.
    """

    template_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    multi_scale: bool = True
    scale_min: float = 0.7
    scale_max: float = 1.3
    scale_steps: int = 10
    ocr_engine: str = "tesseract"
    ocr_lang: str = "eng"
    ocr_config: str = "--psm 6"
    ocr_preprocess: bool = True
    yolo_enabled: bool = False
    yolo_model_path: str | None = None
    yolo_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    screen_state_method: str = "combined"
    screen_state_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    screen_change_mse_threshold: float = Field(default=100.0, ge=0.0)


class MatchingConfig(BaseModel):
    """Настройки сопоставления текста.

    Attributes:
        fuzzy_threshold: Порог оценки RapidFuzz (0–100).
        normalize_whitespace: Сжимать пробелы перед сопоставлением.
        case_insensitive: Сопоставление без учёта регистра.
    """

    fuzzy_threshold: int = Field(default=80, ge=0, le=100)
    normalize_whitespace: bool = True
    case_insensitive: bool = True


class ActionsConfig(BaseModel):
    """Настройки тайм-аутов действий.

    Attributes:
        mouse_move_duration: Длительность плавного перемещения мыши (секунды).
        typing_interval: Интервал между нажатиями клавиш (секунды).
        double_click_interval: Интервал между нажатиями при двойном клике (секунды).
        humanize: Добавлять случайный разброс координат и задержки перед действиями.
        humanize_offset_px: Максимальное случайное смещение (±пиксели) для координат.
        humanize_delay_min_ms: Минимальная случайная задержка перед действием (миллисекунды).
        humanize_delay_max_ms: Максимальная случайная задержка перед действием (миллисекунды).
        clipboard_settle_delay: Ожидание после Ctrl+C перед чтением буфера обмена (секунды).
    """

    mouse_move_duration: float = 0.2
    typing_interval: float = 0.03
    double_click_interval: float = 0.1
    humanize: bool = True
    humanize_offset_px: int = 2
    humanize_delay_min_ms: int = 50
    humanize_delay_max_ms: int = 150
    clipboard_settle_delay: float = 0.2


class LoggingConfig(BaseModel):
    """Настройки логирования.

    Attributes:
        level: Уровень детализации логов (debug/info/warning/error).
        output_dir: Корневой каталог для артефактов запуска.
        save_screenshots: Сохранять скриншоты для каждого шага.
        save_on_failure: Всегда сохранять скриншот при ошибке шага.
        structured: Использовать вывод structlog в формате JSON.
    """

    level: str = "info"
    output_dir: str = "logs/"
    save_screenshots: bool = True
    save_on_failure: bool = True
    structured: bool = True


class LearningConfig(BaseModel):
    """Настройки обучения и кэширования.

    Attributes:
        cache_enabled: Включить кэширование результатов действий.
        cache_path: Путь к JSON-файлу кэша.
        cache_ttl_seconds: Максимальный возраст записи в кэше (секунды).
        pattern_learning: Включить изучение синонимов на основе наблюдений.
    """

    cache_enabled: bool = True
    cache_path: str = "logs/cache.json"
    cache_ttl_seconds: float = 86400.0
    pattern_learning: bool = False


class RetryConfig(BaseModel):
    """Настройки повторных попыток и экспоненциальной задержки.

    Attributes:
        max_attempts: Максимальное количество попыток для каждого шага.
        backoff_base: Множитель экспоненциальной задержки.
        backoff_max: Максимальное время ожидания между попытками (секунды).
    """

    max_attempts: int = 3
    backoff_base: float = 1.5
    backoff_max: float = 30.0


class ReportConfig(BaseModel):
    """Настройки генерации HTML-отчётов.

    Attributes:
        enabled: Автоматически создавать отчёт после каждого запуска.
        output_path: Имя файла отчёта относительно output_dir запуска.
        thumbnail_max_width: Максимальная ширина миниатюры скриншота (пиксели).
        jpeg_quality: Качество JPEG для миниатюр (10–100).
        include_screenshots: Включать ли скриншоты в отчёт.
    """

    enabled: bool = False
    output_path: str = "report.html"
    thumbnail_max_width: int = Field(default=400, ge=50)
    jpeg_quality: int = Field(default=72, ge=10, le=100)
    include_screenshots: bool = True


class AppConfig(BaseModel):
    """Корневая конфигурация приложения.

    Все подсекции имеют разумные значения по умолчанию, поэтому пустой
    файл конфигурации является допустимым.
    """

    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


# ---------------------------------------------------------------------------
# Вспомогательные функции загрузки
# ---------------------------------------------------------------------------

_DEFAULTS_PATH = Path(__file__).parents[2] / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивно объединяет *override* с *base*, возвращая новый словарь.

    Args:
        base: Базовый словарь.
        override: Значения, накладываемые поверх *base*.

    Returns:
        Новый объединённый словарь (исходные словари не изменяются).
    """
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | str | None = None) -> AppConfig:
    """Загружает и валидирует конфигурацию приложения.

    Объединяет слои:
    1. Значения по умолчанию Pydantic
    2. ``config/default.yaml`` (если файл существует)
    3. YAML по указанному *path* (если передан)

    Args:
        path: Необязательный путь к YAML-файлу конфигурации проекта или сценария.

    Returns:
        Валидированный экземпляр :class:`AppConfig`.

    Raises:
        pydantic.ValidationError: Если значение конфигурации не прошло валидацию.
        FileNotFoundError: Если *path* передан, но файл не существует.
    """
    merged: dict[str, Any] = {}

    # Слой 1: встроенный файл значений по умолчанию
    if _DEFAULTS_PATH.exists():
        with _DEFAULTS_PATH.open("r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, defaults)

    # Слой 2: конфигурация, предоставленная пользователем
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, user_cfg)

    return AppConfig(**merged)


def merge_scenario_config(base: AppConfig, scenario_overrides: dict[str, Any]) -> AppConfig:
    """Накладывает переопределения конфигурации сценария поверх базовой конфигурации.

    Args:
        base: Валидированная базовая конфигурация.
        scenario_overrides: Сырой словарь из ключа ``config:`` сценария.

    Returns:
        Новый экземпляр :class:`AppConfig` с применёнными переопределениями.
    """
    base_dict = base.model_dump()
    merged = _deep_merge(base_dict, scenario_overrides)
    return AppConfig(**merged)
