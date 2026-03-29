"""ScreenWalker — фреймворк для RPA на основе компьютерного зрения.

Автоматизирует любое UI-приложение через скриншоты, OCR и template matching —
без доступа к DOM или API-хуков.

Example:
    >>> from screenwalker.core.engine import ScenarioEngine
    >>> engine = ScenarioEngine.from_yaml("scenarios/example_scenario.yaml")
    >>> engine.run()
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
