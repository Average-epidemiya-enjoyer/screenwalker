"""Step logger — records each step execution with JSONL metadata and screenshots.

Each run produces a directory like::

    logs/2024-01-15_14-30-00_create_ticket/
        steps.jsonl           ← one JSON object per line
        screenshots/
            step_001_before.png
            step_001_after.png
            step_002_before.png
            ...

Two complementary APIs are provided:

* :meth:`StepLogger.log_step` — backward-compatible; accepts the engine's
  :class:`~screenwalker.core.context.StepRecord` and extracts vision metadata
  from its ``metadata`` dict.
* :meth:`StepLogger.log_result` — richer API that accepts a fully-populated
  :class:`StepResult` with explicit vision fields.
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
# StepResult — rich step record for JSONL serialisation
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Rich record of a completed step for JSONL logging.

    Attributes:
        step_id: Identifier of the step.
        scenario: Name of the parent scenario.
        action: Action performed (e.g. ``"click"``, ``"type"``).
        success: Whether the step completed without error.
        duration_ms: Wall-clock execution time in milliseconds.
        timestamp: ISO-8601 UTC timestamp of execution.
        find_target: Text / template / label that was searched for.
        method_used: Vision method that located the element
            (``ocr`` / ``template`` / ``yolo`` / ``cache``).
        confidence: Match confidence in ``[0.0, 1.0]``.
        bbox: Bounding box ``(x, y, w, h)`` of the found element.
        found_text: Actual text returned by OCR (may differ from
            *find_target* when fuzzy matching is active).
        error: Error message if the step failed.
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
    """Logs step execution records to a per-run JSONL file with screenshots.

    Attributes:
        output_dir: Root directory for the current run (created lazily).
        save_screenshots: Persist before/after screenshots for every step.
        save_on_failure: Always save a screenshot when a step fails, even if
            *save_screenshots* is ``False``.
    """

    def __init__(
        self,
        output_dir: Path | str,
        save_screenshots: bool = True,
        save_on_failure: bool = True,
    ) -> None:
        """Initialise StepLogger.

        Args:
            output_dir: Per-run output directory (e.g.
                ``logs/2024-01-15_14-30-00_create_ticket``).
            save_screenshots: Save PNG screenshots for every step.
            save_on_failure: Override *save_screenshots* for failed steps.
        """
        self.output_dir = Path(output_dir)
        self.save_screenshots = save_screenshots
        self.save_on_failure = save_on_failure
        self._step_index: int = 0
        self._scenario_name: str = ""

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @property
    def jsonl_path(self) -> Path:
        """Path to the JSONL log file for the current run."""
        return self.output_dir / "steps.jsonl"

    @property
    def screenshots_dir(self) -> Path:
        """Directory where step screenshots are stored."""
        return self.output_dir / "screenshots"

    # ------------------------------------------------------------------
    # Primary API — StepResult
    # ------------------------------------------------------------------

    def log_result(
        self,
        result: StepResult,
        screenshot_before: Image.Image | None = None,
        screenshot_after: Image.Image | None = None,
    ) -> None:
        """Write a JSONL entry and optionally save before/after screenshots.

        Args:
            result: Rich step record to persist.
            screenshot_before: Screenshot taken immediately before the action.
            screenshot_after: Screenshot taken immediately after the action.
        """
        self._step_index += 1
        idx = self._step_index

        # -- Screenshots --------------------------------------------------
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
    # Backward-compatible API — StepRecord
    # ------------------------------------------------------------------

    def log_step(
        self,
        record: StepRecord,
        screenshot: Image.Image | None = None,
    ) -> None:
        """Log a step from a :class:`~screenwalker.core.context.StepRecord`.

        Extracts vision metadata (``find_target``, ``method_used``,
        ``confidence``, ``bbox``, ``found_text``) from
        ``record.metadata`` and delegates to :meth:`log_result`.

        Args:
            record: Engine-produced step execution record.
            screenshot: Optional screenshot captured after the step.
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
    # Scenario-level events
    # ------------------------------------------------------------------

    def log_scenario_start(self, scenario_name: str, variable_count: int) -> None:
        """Record the beginning of a scenario run.

        Args:
            scenario_name: Display name of the scenario.
            variable_count: Number of runtime variables loaded.
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
        """Record the completion of a scenario run.

        Args:
            scenario_name: Display name of the scenario.
            total_steps: Total steps attempted.
            failed_steps: Steps that raised an error.
            elapsed: Total wall-clock time in seconds.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_screenshot(self, image: Image.Image, filename: str) -> Path:
        """Save *image* to the screenshots sub-directory.

        Args:
            image: PIL image to save.
            filename: Destination filename (no path component).

        Returns:
            Absolute path of the saved file.
        """
        dest = self.screenshots_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, "PNG")
        return dest.resolve()

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        """Append *entry* as a single JSON line to the run's JSONL file.

        Args:
            entry: Dict to serialise.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_step_screenshot(
        self, image: Image.Image, step_id: str, success: bool
    ) -> Path:
        """Legacy helper for callers that pass step_id + success separately.

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
        return self._save_screenshot(image, filename)
