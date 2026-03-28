"""Step dataclass — typed description of a single scenario step.

Each step maps to one user action (click, type, assert, etc.) plus an optional
element-finder specification (how to locate the target UI element).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StepAction(str, Enum):
    """Enumeration of all supported step actions."""

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
    """Vision method used to locate a UI element."""

    OCR = "ocr"
    TEMPLATE = "template"
    YOLO = "yolo"


class OnFailure(str, Enum):
    """Behaviour when a step fails."""

    ABORT = "abort"      # Stop the scenario immediately (default)
    SKIP = "skip"        # Log the failure and continue to the next step
    CONTINUE = "continue"  # Alias for skip; kept for readability in YAML


class FindSpec(BaseModel):
    """Specification for locating a UI element before acting on it.

    Attributes:
        method: Vision method to use (ocr, template, yolo).
        query: Text to search via OCR, or path to template image.
        template: Explicit template image path (alternative to query for template method).
        threshold: Minimum confidence score (0.0–1.0).
        region: Named region or [x, y, w, h] bounding box to restrict search.
        offset: [dx, dy] pixel offset applied to the found element centre.
        fuzzy: Whether to use fuzzy text matching for OCR results.
        fuzzy_threshold: RapidFuzz score threshold (0–100).
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
        """Ensure offset is exactly two integers if provided."""
        if v is not None and len(v) != 2:
            raise ValueError("offset must be a list of exactly 2 integers [dx, dy]")
        return v


class Step(BaseModel):
    """A single step in a scenario.

    Attributes:
        id: Unique identifier for this step within the scenario.
        description: Human-readable description shown in logs.
        action: The action to perform.
        find: How to locate the target element (optional for some actions).
        fallback: Alternative finder if the primary find fails.
        text: Text to type (for ``type`` action).
        keys: Key combination to press (for ``hotkey`` action).
        target: Application path / URL (for ``launch`` action).
        label: Screenshot label (for ``screenshot`` action).
        wait: Duration in seconds (for ``wait`` action).
        wait_after: Pause in seconds after action executes.
        timeout: Override the global step timeout for this step.
        on_failure: Behaviour when this step fails.
        clear_first: Select-all + delete before typing (for ``type`` action).
        extra: Arbitrary extra keys passed through from YAML.
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

    model_config = {"extra": "allow"}

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, v: list[str] | None) -> list[str] | None:
        """Ensure key names are non-empty strings."""
        if v is not None:
            for key in v:
                if not key.strip():
                    raise ValueError("Key names must be non-empty strings")
        return v
