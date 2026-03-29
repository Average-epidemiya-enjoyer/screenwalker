"""Configuration loading and Pydantic validation.

The configuration is layered:
1. Built-in defaults (hardcoded in Pydantic models).
2. ``config/default.yaml`` merged on top.
3. Project-level YAML passed via ``--config``.
4. Per-scenario ``config:`` block in the scenario YAML.

Later layers override earlier ones.  All keys are validated by Pydantic so
typos in YAML surface at startup, not mid-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class TimeoutsConfig(BaseModel):
    """Timing configuration.

    Attributes:
        step_default: Default per-step timeout in seconds.
        poll_interval: Interval between find-element retries.
        screenshot_delay: Pause before capturing (UI settle time).
        action_delay: Pause after each action.
    """

    step_default: float = 15.0
    poll_interval: float = 0.5
    screenshot_delay: float = 0.3
    action_delay: float = 0.1


class VisionConfig(BaseModel):
    """Vision pipeline configuration.

    Attributes:
        template_threshold: Minimum template match confidence.
        multi_scale: Enable multi-scale template search.
        scale_min: Minimum scale factor for multi-scale search.
        scale_max: Maximum scale factor for multi-scale search.
        scale_steps: Number of scale levels.
        ocr_engine: OCR backend — ``"tesseract"`` or ``"paddleocr"``.
        ocr_lang: Tesseract language code(s).
        ocr_config: Tesseract CLI flags.
        ocr_preprocess: Pre-process images before OCR.
        yolo_enabled: Enable YOLO detection.
        yolo_model_path: Path to YOLO weights.
        yolo_confidence: Minimum YOLO detection confidence.
        screen_state_method: Screen-state identification strategy.
        screen_state_threshold: Minimum confidence for state identification.
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
    """Text matching configuration.

    Attributes:
        fuzzy_threshold: RapidFuzz score threshold (0–100).
        normalize_whitespace: Collapse whitespace before matching.
        case_insensitive: Case-insensitive matching.
    """

    fuzzy_threshold: int = Field(default=80, ge=0, le=100)
    normalize_whitespace: bool = True
    case_insensitive: bool = True


class ActionsConfig(BaseModel):
    """Action timing configuration.

    Attributes:
        mouse_move_duration: Smooth mouse movement duration in seconds.
        typing_interval: Seconds between keystrokes.
        double_click_interval: Seconds between double-click presses.
        humanize: Add random jitter to coordinates and pre-action delays.
        humanize_offset_px: Maximum random pixel offset (±) applied to coordinates.
        humanize_delay_min_ms: Minimum random pre-action delay in milliseconds.
        humanize_delay_max_ms: Maximum random pre-action delay in milliseconds.
        clipboard_settle_delay: Seconds to wait after Ctrl+C before reading clipboard.
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
    """Logging configuration.

    Attributes:
        level: Log verbosity (debug/info/warning/error).
        output_dir: Root directory for run artefacts.
        save_screenshots: Persist screenshots per step.
        save_on_failure: Always save screenshot on step failure.
        structured: Use structlog JSON output.
    """

    level: str = "info"
    output_dir: str = "logs/"
    save_screenshots: bool = True
    save_on_failure: bool = True
    structured: bool = True


class LearningConfig(BaseModel):
    """Learning / caching configuration.

    Attributes:
        cache_enabled: Enable action result caching.
        cache_path: Path to the JSON cache file.
        cache_ttl_seconds: Maximum age of a cache entry.
        pattern_learning: Enable synonym learning from observations.
    """

    cache_enabled: bool = True
    cache_path: str = "logs/cache.json"
    cache_ttl_seconds: float = 86400.0
    pattern_learning: bool = False


class RetryConfig(BaseModel):
    """Retry and backoff configuration.

    Attributes:
        max_attempts: Maximum retry attempts per step.
        backoff_base: Exponential backoff multiplier.
        backoff_max: Maximum wait between retries in seconds.
    """

    max_attempts: int = 3
    backoff_base: float = 1.5
    backoff_max: float = 30.0


class AppConfig(BaseModel):
    """Root application configuration.

    All sub-sections have sensible defaults so an empty config file is valid.
    """

    timeouts: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_DEFAULTS_PATH = Path(__file__).parents[2] / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Args:
        base: Base dictionary.
        override: Values to overlay on *base*.

    Returns:
        New merged dictionary (neither input is mutated).
    """
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate application configuration.

    Merges:
    1. Pydantic defaults
    2. ``config/default.yaml`` (if present)
    3. *path* YAML (if provided)

    Args:
        path: Optional path to a project or scenario config YAML.

    Returns:
        Validated :class:`AppConfig` instance.

    Raises:
        pydantic.ValidationError: If any config value fails validation.
        FileNotFoundError: If *path* is provided but does not exist.
    """
    merged: dict[str, Any] = {}

    # Layer 1: built-in defaults file
    if _DEFAULTS_PATH.exists():
        with _DEFAULTS_PATH.open("r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, defaults)

    # Layer 2: user-provided config
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, user_cfg)

    return AppConfig(**merged)


def merge_scenario_config(base: AppConfig, scenario_overrides: dict[str, Any]) -> AppConfig:
    """Merge per-scenario config overrides on top of a base config.

    Args:
        base: Validated base configuration.
        scenario_overrides: Raw dict from the scenario's ``config:`` key.

    Returns:
        New :class:`AppConfig` with overrides applied.
    """
    base_dict = base.model_dump()
    merged = _deep_merge(base_dict, scenario_overrides)
    return AppConfig(**merged)
