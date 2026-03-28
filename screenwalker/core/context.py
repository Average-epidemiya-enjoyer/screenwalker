"""RunContext — mutable runtime state for a single scenario execution.

The context is threaded through every step and vision call so that all
components share the same variables, history, and screenshots without
tight coupling to each other.
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
    """Immutable record of a completed step execution.

    Attributes:
        step_id: Identifier of the step.
        action: Action string (e.g. "click").
        success: Whether the step completed without error.
        elapsed: Wall-clock time in seconds.
        screenshot_path: Path to the screenshot taken after this step (if any).
        error: Error message if the step failed.
        metadata: Arbitrary key-value pairs for debugging.
    """

    step_id: str
    action: str
    success: bool
    elapsed: float
    screenshot_path: Path | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RunContext:
    """Holds all mutable state for a single scenario run.

    The context is intentionally *not* a frozen dataclass — steps need to
    update variables, push history records, and cache screenshots during
    execution.

    Attributes:
        scenario_name: Human-readable scenario name from YAML.
        run_id: Unique run identifier (timestamp-based).
        variables: Mutable variable bag interpolated into step text/queries.
        history: Ordered list of step execution records.
        current_screen: Identifier of the last detected screen state.
        last_screenshot: Most recent captured PIL image.
        output_dir: Directory where logs and screenshots are written.
        dry_run: When True, actions are logged but not executed.
    """

    def __init__(
        self,
        scenario_name: str,
        variables: dict[str, str] | None = None,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize RunContext.

        Args:
            scenario_name: Name of the scenario being executed.
            variables: Initial variable dictionary (from YAML + CLI overrides).
            output_dir: Root directory for logs and screenshots.
            dry_run: Whether to skip actual action execution.
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
    # Variable interpolation
    # ------------------------------------------------------------------

    def interpolate(self, text: str) -> str:
        """Replace ``{{ variable_name }}`` placeholders in *text*.

        Args:
            text: Template string potentially containing ``{{ var }}`` tokens.

        Returns:
            String with all known variables substituted. Unknown variables
            are left as-is.

        Example:
            >>> ctx = RunContext("demo", variables={"user": "admin"})
            >>> ctx.interpolate("Hello {{ user }}!")
            'Hello admin!'
        """
        # TODO: implement Jinja2-style substitution using self.variables
        def replace(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            return self.variables.get(key, match.group(0))

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, text)

    def set_variable(self, key: str, value: str) -> None:
        """Set or update a runtime variable.

        Args:
            key: Variable name.
            value: New value (always stored as a string).
        """
        self.variables[key] = value

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def record_step(self, record: StepRecord) -> None:
        """Append a completed step record to the execution history.

        Args:
            record: The :class:`StepRecord` to append.
        """
        self.history.append(record)

    @property
    def failed_steps(self) -> list[StepRecord]:
        """Return all step records that represent failures."""
        return [r for r in self.history if not r.success]

    @property
    def step_count(self) -> int:
        """Total number of executed steps (successful + failed)."""
        return len(self.history)

    # ------------------------------------------------------------------
    # Screenshot helpers
    # ------------------------------------------------------------------

    def save_screenshot(self, image: Image.Image, label: str = "step") -> Path:
        """Persist a PIL image to the run output directory.

        Args:
            image: PIL image to save.
            label: Filename label (step id or description slug).

        Returns:
            Absolute path of the saved file.
        """
        # TODO: ensure output_dir exists, save image, return path
        raise NotImplementedError("TODO: implement screenshot persistence")

    def update_screenshot(self, image: Image.Image) -> None:
        """Update the cached last screenshot.

        Args:
            image: The latest captured screen image.
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
