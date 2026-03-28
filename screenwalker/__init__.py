"""ScreenWalker — vision-based RPA framework.

Automate any UI application by interacting with it through screenshots,
OCR, and template matching — no DOM access or API hooks required.

Example:
    >>> from screenwalker.core.engine import ScenarioEngine
    >>> engine = ScenarioEngine.from_yaml("scenarios/example_scenario.yaml")
    >>> engine.run()
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
