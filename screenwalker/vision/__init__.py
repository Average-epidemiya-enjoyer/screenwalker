"""Vision layer — screen capture, OCR, template matching, and element detection.

All public finders implement the :class:`Finder` Protocol and return a
:class:`FindResult` so the engine can treat them interchangeably.
"""

from screenwalker.vision.screen_state import FindResult

__all__ = ["FindResult"]
