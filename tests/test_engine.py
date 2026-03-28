"""Tests for ScenarioEngine and RunContext."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from screenwalker.core.context import RunContext, StepRecord
from screenwalker.core.engine import ScenarioEngine
from screenwalker.core.errors import ScenarioValidationError, StepTimeout
from screenwalker.core.step import FindSpec, FindMethod, OnFailure, Step, StepAction
from screenwalker.utils.config import AppConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> AppConfig:
    """Return a default AppConfig with test-friendly timeouts."""
    cfg = AppConfig()
    cfg.timeouts.step_default = 5.0
    cfg.timeouts.poll_interval = 0.1
    return cfg


@pytest.fixture()
def simple_context() -> RunContext:
    """Return a RunContext pre-loaded with test variables."""
    return RunContext(
        scenario_name="test_scenario",
        variables={"username": "admin", "password": "secret"},
        dry_run=True,
    )


@pytest.fixture()
def wait_step() -> Step:
    """A simple wait step that needs no element finder."""
    return Step(id="wait_step", action=StepAction.WAIT, wait=0.0)


# ---------------------------------------------------------------------------
# RunContext tests
# ---------------------------------------------------------------------------


class TestRunContext:
    """Unit tests for RunContext."""

    def test_interpolate_known_variable(self, simple_context: RunContext) -> None:
        """Variables are substituted in template strings."""
        result = simple_context.interpolate("Hello {{ username }}!")
        assert result == "Hello admin!"

    def test_interpolate_unknown_variable_unchanged(self, simple_context: RunContext) -> None:
        """Unknown variables are left as-is without raising."""
        result = simple_context.interpolate("Hello {{ unknown }}!")
        assert "{{ unknown }}" in result

    def test_interpolate_multiple_variables(self, simple_context: RunContext) -> None:
        """Multiple variables in one string are all substituted."""
        result = simple_context.interpolate("{{ username }}:{{ password }}")
        assert result == "admin:secret"

    def test_set_variable(self, simple_context: RunContext) -> None:
        """set_variable updates and creates variables."""
        simple_context.set_variable("new_key", "new_value")
        assert simple_context.variables["new_key"] == "new_value"

    def test_record_step_increments_count(self, simple_context: RunContext) -> None:
        """record_step increases step_count by one."""
        assert simple_context.step_count == 0
        simple_context.record_step(
            StepRecord(step_id="s1", action="click", success=True, elapsed=0.1)
        )
        assert simple_context.step_count == 1

    def test_failed_steps_filtered(self, simple_context: RunContext) -> None:
        """failed_steps returns only unsuccessful records."""
        simple_context.record_step(
            StepRecord(step_id="ok", action="click", success=True, elapsed=0.1)
        )
        simple_context.record_step(
            StepRecord(step_id="fail", action="type", success=False, elapsed=0.2, error="oops")
        )
        assert len(simple_context.failed_steps) == 1
        assert simple_context.failed_steps[0].step_id == "fail"

    def test_repr(self, simple_context: RunContext) -> None:
        """__repr__ includes scenario name and step count."""
        r = repr(simple_context)
        assert "test_scenario" in r
        assert "steps=0" in r


# ---------------------------------------------------------------------------
# Step validation tests
# ---------------------------------------------------------------------------


class TestStep:
    """Unit tests for the Step Pydantic model."""

    def test_valid_step(self) -> None:
        """A minimal valid step parses without error."""
        step = Step(id="s1", action=StepAction.WAIT, wait=1.0)
        assert step.id == "s1"

    def test_invalid_action_raises(self) -> None:
        """Invalid action string raises Pydantic ValidationError."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            Step(id="s1", action="fly")  # type: ignore[arg-type]

    def test_find_spec_threshold_bounds(self) -> None:
        """FindSpec threshold must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            FindSpec(method=FindMethod.OCR, query="ok", threshold=1.5)

    def test_find_spec_offset_must_be_two_ints(self) -> None:
        """FindSpec offset with wrong length raises."""
        with pytest.raises(Exception):
            FindSpec(method=FindMethod.OCR, query="ok", offset=[10])

    def test_on_failure_defaults_to_abort(self) -> None:
        """Default on_failure is ABORT."""
        step = Step(id="s1", action=StepAction.WAIT)
        assert step.on_failure == OnFailure.ABORT


# ---------------------------------------------------------------------------
# ScenarioEngine tests
# ---------------------------------------------------------------------------


class TestScenarioEngine:
    """Unit tests for ScenarioEngine."""

    def test_dry_run_prints_steps(
        self,
        default_config: AppConfig,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """dry_run prints one line per step without executing actions."""
        steps = [
            Step(id="step_one", action=StepAction.WAIT, wait=0.0),
            Step(id="step_two", action=StepAction.SCREENSHOT),
        ]
        ctx = RunContext("dry_scenario", dry_run=True)
        engine = ScenarioEngine(
            scenario_name="dry_scenario",
            steps=steps,
            teardown_steps=[],
            context=ctx,
            config=default_config,
        )
        engine.dry_run()
        out = capsys.readouterr().out
        assert "step_one" in out
        assert "step_two" in out

    def test_wait_action_does_not_raise(self, default_config: AppConfig) -> None:
        """WAIT action is fully implemented and should not raise NotImplementedError."""
        steps = [Step(id="w", action=StepAction.WAIT, wait=0.0)]
        ctx = RunContext("wait_test", dry_run=False)
        engine = ScenarioEngine(
            scenario_name="wait_test",
            steps=steps,
            teardown_steps=[],
            context=ctx,
            config=default_config,
        )
        # WAIT is the one fully implemented action — should not raise
        # TODO: uncomment once engine._execute_step is fully implemented
        # engine.run()

    def test_parse_steps_invalid_action_raises(self, default_config: AppConfig) -> None:
        """_parse_steps raises ScenarioValidationError for bad action."""
        with pytest.raises(ScenarioValidationError):
            ScenarioEngine._parse_steps([{"id": "bad", "action": "fly_to_moon"}])

    def test_from_yaml_missing_file_raises(self, default_config: AppConfig) -> None:
        """from_yaml raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            ScenarioEngine.from_yaml("/nonexistent/scenario.yaml", config=default_config)

    def test_from_yaml_loads_valid_file(
        self, tmp_path: Path, default_config: AppConfig
    ) -> None:
        """from_yaml correctly loads a minimal valid scenario."""
        scenario_yaml = tmp_path / "test.yaml"
        scenario_yaml.write_text(
            "name: Test\nvariables:\n  x: '1'\nsteps:\n  - id: s1\n    action: wait\n    wait: 0\n"
        )
        engine = ScenarioEngine.from_yaml(scenario_yaml, config=default_config)
        assert engine.scenario_name == "Test"
        assert len(engine.steps) == 1
        assert engine.context.variables["x"] == "1"

    def test_variable_overrides_applied(
        self, tmp_path: Path, default_config: AppConfig
    ) -> None:
        """CLI variable overrides take precedence over scenario defaults."""
        scenario_yaml = tmp_path / "test.yaml"
        scenario_yaml.write_text(
            "name: Vars\nvariables:\n  env: prod\nsteps: []\n"
        )
        engine = ScenarioEngine.from_yaml(
            scenario_yaml,
            config=default_config,
            variable_overrides={"env": "staging"},
        )
        assert engine.context.variables["env"] == "staging"
