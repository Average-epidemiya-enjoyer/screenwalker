"""ScenarioEngine — main orchestrator implementing a step-execution state machine.

The engine loads a scenario from YAML, validates it, and drives each step
through the vision → action → assert pipeline while managing retries,
fallbacks, and the :class:`~screenwalker.core.context.RunContext`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog
import yaml

from screenwalker.core.context import RunContext, StepRecord
from screenwalker.core.errors import (
    ElementNotFound,
    ScenarioError,
    ScenarioValidationError,
    StepTimeout,
)
from screenwalker.core.step import OnFailure
from screenwalker.core.step import Step, StepAction
from screenwalker.utils.config import AppConfig

logger = structlog.get_logger(__name__)


class ScenarioEngine:
    """Executes a scenario YAML as an ordered state machine.

    Each step is processed in sequence:

    1. **Interpolate** — substitute ``{{ variables }}`` in queries and text.
    2. **Locate** — run the configured vision finder (OCR / template / YOLO).
    3. **Act** — delegate to the appropriate action handler.
    4. **Record** — write a :class:`~screenwalker.core.context.StepRecord`.
    5. **Retry / fallback** — on failure, try fallback finder or retry up to
       ``config.retry.max_attempts`` times with exponential back-off.

    Attributes:
        scenario_name: Human-readable name from YAML.
        steps: Ordered list of :class:`~screenwalker.core.step.Step` objects.
        teardown_steps: Steps executed after main steps (even on failure).
        context: Mutable :class:`~screenwalker.core.context.RunContext`.
        config: Validated :class:`~screenwalker.utils.config.AppConfig`.
    """

    def __init__(
        self,
        scenario_name: str,
        steps: list[Step],
        teardown_steps: list[Step],
        context: RunContext,
        config: AppConfig,
    ) -> None:
        """Initialize ScenarioEngine.

        Args:
            scenario_name: Scenario display name.
            steps: Main execution steps.
            teardown_steps: Steps run after main steps regardless of outcome.
            context: Run context shared across all steps.
            config: Validated application configuration.
        """
        self.scenario_name = scenario_name
        self.steps = steps
        self.teardown_steps = teardown_steps
        self.context = context
        self.config = config
        self._log = structlog.get_logger(__name__).bind(scenario=scenario_name)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        config: AppConfig | None = None,
        variable_overrides: dict[str, str] | None = None,
    ) -> "ScenarioEngine":
        """Load and validate a scenario YAML file, returning a ready engine.

        Args:
            path: Path to the scenario YAML file.
            config: Pre-loaded application config; loads defaults if None.
            variable_overrides: CLI ``--var`` overrides merged on top of
                scenario ``variables`` section.

        Returns:
            A fully initialised :class:`ScenarioEngine`.

        Raises:
            ScenarioValidationError: If the YAML is invalid or missing required fields.
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ScenarioValidationError("Scenario YAML must be a mapping at top level")

        # TODO: validate with Pydantic schema
        scenario_name: str = raw.get("name", path.stem)
        raw_variables: dict[str, str] = raw.get("variables", {})
        variables = {**raw_variables, **(variable_overrides or {})}

        # TODO: load config overrides from raw.get("config") and merge
        from screenwalker.utils.config import load_config

        resolved_config = config or load_config(None)

        # Parse steps
        steps = cls._parse_steps(raw.get("steps", []))
        teardown_steps = cls._parse_steps(raw.get("teardown", []))

        context = RunContext(
            scenario_name=scenario_name,
            variables=variables,
            output_dir=Path(resolved_config.logging.output_dir) / scenario_name,
        )

        return cls(
            scenario_name=scenario_name,
            steps=steps,
            teardown_steps=teardown_steps,
            context=context,
            config=resolved_config,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all scenario steps, then run teardown steps.

        Raises:
            ScenarioError: Re-raised after teardown if any step with
                ``on_failure=abort`` fails.
        """
        self._log.info("Scenario started", step_count=len(self.steps))
        failure: ScenarioError | None = None

        try:
            for step in self.steps:
                try:
                    self._execute_step(step)
                except ScenarioError as exc:
                    if step.on_failure == OnFailure.ABORT:
                        failure = exc
                        break
                    self._log.warning("Step failed, continuing", step_id=step.id, error=str(exc))
        finally:
            self._run_teardown()

        if failure:
            raise failure

        self._log.info(
            "Scenario finished",
            total=self.context.step_count,
            failed=len(self.context.failed_steps),
        )

    def dry_run(self) -> None:
        """Print planned steps without executing any actions."""
        self._log.info("DRY RUN — no actions will be executed")
        for i, step in enumerate(self.steps, start=1):
            click_repr = f"[{step.action.value}]"
            find_repr = f"find={step.find.method.value}:{step.find.query!r}" if step.find else ""
            print(f"  {i:>3}. {step.id:<30} {click_repr:<20} {find_repr}")

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _execute_step(self, step: Step) -> None:
        """Execute a single step through the full pipeline.

        Args:
            step: The step to execute.

        Raises:
            StepTimeout: If the element is not found within the timeout.
            ElementNotFound: If all finders are exhausted.
            ActionError: If the action itself raises.
        """
        log = self._log.bind(step_id=step.id, action=step.action.value)
        log.info("Executing step", description=step.description)

        start = time.monotonic()
        success = False
        error: str | None = None

        try:
            # Interpolate variables into text / query fields
            interpolated_step = self._interpolate_step(step)

            # Locate element (if a find spec is present)
            find_result = None
            if interpolated_step.find:
                find_result = self._locate_with_retry(interpolated_step)

            # Dispatch action
            self._dispatch_action(interpolated_step, find_result)

            if step.wait_after > 0:
                time.sleep(step.wait_after)

            success = True

        except ScenarioError as exc:
            error = str(exc)
            raise
        finally:
            elapsed = time.monotonic() - start
            self.context.record_step(
                StepRecord(
                    step_id=step.id,
                    action=step.action.value,
                    success=success,
                    elapsed=elapsed,
                    error=error,
                )
            )

    def _interpolate_step(self, step: Step) -> Step:
        """Return a copy of *step* with all ``{{ var }}`` tokens resolved.

        Args:
            step: Original step (not mutated).

        Returns:
            New step with variables substituted.
        """
        # TODO: deep-copy step and interpolate all string fields
        return step

    def _locate_with_retry(self, step: Step) -> Any:
        """Attempt to locate the element described by step.find, with retries.

        Args:
            step: Step containing a populated ``find`` spec.

        Returns:
            A :class:`~screenwalker.vision.screen_state.FindResult`.

        Raises:
            StepTimeout: If the element is not found within the timeout.
        """
        # TODO: import vision finders, poll until found or timeout
        raise NotImplementedError("TODO: implement locate_with_retry")

    def _dispatch_action(self, step: Step, find_result: Any) -> None:
        """Route a step to the appropriate action handler.

        Args:
            step: The (interpolated) step to execute.
            find_result: Located element, or None for actions that don't need one.
        """
        # TODO: import action modules and dispatch based on step.action
        if self.context.dry_run:
            logger.debug("DRY RUN action skipped", action=step.action.value, step_id=step.id)
            return

        action_map = {
            StepAction.CLICK: self._action_click,
            StepAction.DOUBLE_CLICK: self._action_double_click,
            StepAction.RIGHT_CLICK: self._action_right_click,
            StepAction.TYPE: self._action_type,
            StepAction.HOTKEY: self._action_hotkey,
            StepAction.SCROLL: self._action_scroll,
            StepAction.DRAG: self._action_drag,
            StepAction.COPY: self._action_copy,
            StepAction.PASTE: self._action_paste,
            StepAction.ASSERT_VISIBLE: self._action_assert_visible,
            StepAction.ASSERT_TEXT: self._action_assert_text,
            StepAction.WAIT: self._action_wait,
            StepAction.LAUNCH: self._action_launch,
            StepAction.SCREENSHOT: self._action_screenshot,
        }

        handler = action_map.get(step.action)
        if handler is None:
            raise NotImplementedError(f"No handler for action: {step.action}")
        handler(step, find_result)

    # ------------------------------------------------------------------
    # Action stubs — each delegates to the relevant actions/ module
    # ------------------------------------------------------------------

    def _action_click(self, step: Step, find_result: Any) -> None:
        """Left-click the located element."""
        # TODO: from screenwalker.actions.mouse import click; click(find_result.bbox)
        raise NotImplementedError("TODO: implement click action")

    def _action_double_click(self, step: Step, find_result: Any) -> None:
        """Double-click the located element."""
        raise NotImplementedError("TODO: implement double_click action")

    def _action_right_click(self, step: Step, find_result: Any) -> None:
        """Right-click the located element."""
        raise NotImplementedError("TODO: implement right_click action")

    def _action_type(self, step: Step, find_result: Any) -> None:
        """Type text, optionally clearing the field first."""
        raise NotImplementedError("TODO: implement type action")

    def _action_hotkey(self, step: Step, find_result: Any) -> None:
        """Send a key combination."""
        raise NotImplementedError("TODO: implement hotkey action")

    def _action_scroll(self, step: Step, find_result: Any) -> None:
        """Scroll at the element position."""
        raise NotImplementedError("TODO: implement scroll action")

    def _action_drag(self, step: Step, find_result: Any) -> None:
        """Drag from element to a target position."""
        raise NotImplementedError("TODO: implement drag action")

    def _action_copy(self, step: Step, find_result: Any) -> None:
        """Copy selection to a context variable via clipboard."""
        raise NotImplementedError("TODO: implement copy action")

    def _action_paste(self, step: Step, find_result: Any) -> None:
        """Paste clipboard text at the current position."""
        raise NotImplementedError("TODO: implement paste action")

    def _action_assert_visible(self, step: Step, find_result: Any) -> None:
        """Assert that an element is visible (find_result must not be None)."""
        raise NotImplementedError("TODO: implement assert_visible action")

    def _action_assert_text(self, step: Step, find_result: Any) -> None:
        """Assert OCR text at location equals expected value."""
        raise NotImplementedError("TODO: implement assert_text action")

    def _action_wait(self, step: Step, find_result: Any) -> None:
        """Sleep for step.wait seconds."""
        duration = step.wait or 0.0
        logger.debug("Waiting", seconds=duration, step_id=step.id)
        time.sleep(duration)

    def _action_launch(self, step: Step, find_result: Any) -> None:
        """Launch an application."""
        raise NotImplementedError("TODO: implement launch action")

    def _action_screenshot(self, step: Step, find_result: Any) -> None:
        """Capture and save a screenshot."""
        raise NotImplementedError("TODO: implement screenshot action")

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _run_teardown(self) -> None:
        """Execute teardown steps, swallowing errors to avoid masking main failure."""
        for step in self.teardown_steps:
            try:
                self._execute_step(step)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Teardown step failed", step_id=step.id, error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_steps(raw_steps: list[dict[str, Any]]) -> list[Step]:
        """Parse a list of raw dicts into typed Step objects.

        Args:
            raw_steps: List of step dicts from YAML.

        Returns:
            List of validated :class:`~screenwalker.core.step.Step` instances.

        Raises:
            ScenarioValidationError: If any step fails Pydantic validation.
        """
        steps: list[Step] = []
        for i, raw in enumerate(raw_steps):
            if not isinstance(raw, dict):
                raise ScenarioValidationError(f"Step at index {i} must be a mapping")
            try:
                steps.append(Step(**raw))
            except Exception as exc:
                step_id = raw.get("id", f"<index {i}>")
                raise ScenarioValidationError(
                    f"Validation error in step {step_id!r}: {exc}"
                ) from exc
        return steps
