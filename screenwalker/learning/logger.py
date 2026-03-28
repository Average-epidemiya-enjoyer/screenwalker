"""Step logger — records each step execution with screenshots and metadata.

Writes structured JSON log entries (via structlog) and optionally saves
before/after screenshots alongside each step record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

from screenwalker.core.context import StepRecord
from screenwalker.vision.capture import save_image

_log = structlog.get_logger(__name__)


class StepLogger:
    """Logs step execution records with optional screenshot capture.

    Attributes:
        output_dir: Root directory for all run artefacts.
        save_screenshots: Whether to persist screenshots per step.
        save_on_failure: Always save screenshot when a step fails.
    """

    def __init__(
        self,
        output_dir: Path | str,
        save_screenshots: bool = True,
        save_on_failure: bool = True,
    ) -> None:
        """Initialize StepLogger.

        Args:
            output_dir: Directory where logs and screenshots are written.
            save_screenshots: Persist a screenshot for every step when True.
            save_on_failure: Always save a screenshot on step failure, even if
                save_screenshots is False.
        """
        self.output_dir = Path(output_dir)
        self.save_screenshots = save_screenshots
        self.save_on_failure = save_on_failure
        self._step_index = 0

    def log_step(
        self,
        record: StepRecord,
        screenshot: Image.Image | None = None,
    ) -> None:
        """Write a structured log entry for a completed step.

        Args:
            record: Execution record from the engine.
            screenshot: Optional screenshot taken after the step.
        """
        self._step_index += 1
        screenshot_path: str | None = None

        should_save = (self.save_screenshots and screenshot) or (
            not record.success and self.save_on_failure and screenshot
        )
        if should_save and screenshot:
            screenshot_path = str(
                self._save_step_screenshot(screenshot, record.step_id, record.success)
            )

        level = "info" if record.success else "warning"
        getattr(_log, level)(
            "Step executed",
            step_id=record.step_id,
            action=record.action,
            success=record.success,
            elapsed=round(record.elapsed, 3),
            error=record.error,
            screenshot=screenshot_path,
            index=self._step_index,
            ts=datetime.now(timezone.utc).isoformat(),
        )

    def log_scenario_start(self, scenario_name: str, variable_count: int) -> None:
        """Log the beginning of a scenario run.

        Args:
            scenario_name: Display name of the scenario.
            variable_count: Number of runtime variables loaded.
        """
        _log.info(
            "Scenario started",
            scenario=scenario_name,
            variables=variable_count,
            ts=datetime.now(timezone.utc).isoformat(),
        )

    def log_scenario_end(
        self,
        scenario_name: str,
        total_steps: int,
        failed_steps: int,
        elapsed: float,
    ) -> None:
        """Log the completion (or failure) of a scenario run.

        Args:
            scenario_name: Display name of the scenario.
            total_steps: Total number of steps attempted.
            failed_steps: Number of steps that raised an error.
            elapsed: Total wall-clock time in seconds.
        """
        success = failed_steps == 0
        level = "info" if success else "error"
        getattr(_log, level)(
            "Scenario ended",
            scenario=scenario_name,
            total_steps=total_steps,
            failed_steps=failed_steps,
            elapsed=round(elapsed, 2),
            success=success,
            ts=datetime.now(timezone.utc).isoformat(),
        )

    def _save_step_screenshot(
        self, image: Image.Image, step_id: str, success: bool
    ) -> Path:
        """Save a step screenshot to the run output directory.

        Args:
            image: PIL screenshot.
            step_id: Step identifier used in the filename.
            success: Whether the step succeeded (affects filename prefix).

        Returns:
            Path of the saved file.
        """
        prefix = "ok" if success else "fail"
        slug = step_id.replace(" ", "_").replace("/", "-")
        filename = f"{self._step_index:03d}_{prefix}_{slug}.png"
        dest = self.output_dir / "screenshots" / filename
        from screenwalker.vision.capture import save_image
        return save_image(image, dest)
