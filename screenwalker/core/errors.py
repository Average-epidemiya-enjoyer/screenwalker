"""Custom exceptions for the ScreenWalker runtime.

All exceptions inherit from :class:`ScenarioError` so callers can catch the
entire hierarchy with a single ``except ScenarioError`` clause while still
being able to handle specific failure modes.
"""

from __future__ import annotations


class ScenarioError(Exception):
    """Base class for all ScreenWalker runtime errors."""

    def __init__(self, message: str, step_id: str | None = None) -> None:
        """Initialize ScenarioError.

        Args:
            message: Human-readable description of the failure.
            step_id: Optional ID of the step that triggered the error.
        """
        super().__init__(message)
        self.step_id = step_id

    def __str__(self) -> str:
        if self.step_id:
            return f"[step={self.step_id}] {super().__str__()}"
        return super().__str__()


class StepTimeout(ScenarioError):
    """Raised when an element is not found within the configured timeout.

    Attributes:
        step_id: ID of the step that timed out.
        timeout: Configured timeout in seconds.
        query: Element query that failed to resolve.
    """

    def __init__(self, step_id: str, timeout: float, query: str) -> None:
        """Initialize StepTimeout.

        Args:
            step_id: ID of the timed-out step.
            timeout: Timeout duration in seconds.
            query: The element query (text, template name, etc.) that was searched.
        """
        super().__init__(
            f"Element not found within {timeout}s: {query!r}",
            step_id=step_id,
        )
        self.timeout = timeout
        self.query = query


class ElementNotFound(ScenarioError):
    """Raised when an element cannot be located after exhausting all finders.

    Attributes:
        step_id: ID of the step that could not find the element.
        query: Element query that was attempted.
        methods: Vision methods that were tried.
    """

    def __init__(self, step_id: str, query: str, methods: list[str]) -> None:
        """Initialize ElementNotFound.

        Args:
            step_id: ID of the step that failed.
            query: The element query string or template name.
            methods: List of vision methods tried (e.g. ['ocr', 'template']).
        """
        methods_str = ", ".join(methods)
        super().__init__(
            f"Element {query!r} not found. Methods tried: [{methods_str}]",
            step_id=step_id,
        )
        self.query = query
        self.methods = methods


class ScreenMismatch(ScenarioError):
    """Raised when the current screen state does not match the expected state.

    Attributes:
        step_id: ID of the step that detected the mismatch.
        expected: Expected screen identifier.
        actual: Actual screen identifier detected (or None if unknown).
    """

    def __init__(self, step_id: str, expected: str, actual: str | None) -> None:
        """Initialize ScreenMismatch.

        Args:
            step_id: ID of the failing step.
            expected: Expected screen state identifier.
            actual: Detected screen state identifier, or None if unrecognized.
        """
        actual_desc = actual if actual else "unknown"
        super().__init__(
            f"Screen mismatch: expected {expected!r}, got {actual_desc!r}",
            step_id=step_id,
        )
        self.expected = expected
        self.actual = actual


class ScenarioValidationError(ScenarioError):
    """Raised when a scenario YAML fails schema validation before execution."""


class ActionError(ScenarioError):
    """Raised when a low-level action (mouse click, key press, etc.) fails."""


class ConfigurationError(ScenarioError):
    """Raised when required configuration is missing or invalid."""
