"""YOLO-based UI element detector.

Provides a :class:`YOLODetector` that wraps the Ultralytics YOLO API and
exposes the same :class:`~screenwalker.vision.screen_state.Finder` Protocol
interface as the OCR and template modules.

Requires the ``yolo`` extra: ``pip install screenwalker[yolo]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from screenwalker.vision.screen_state import BBox, FindResult


class YOLODetector:
    """Detect UI elements using a YOLO model.

    This class is a **stub** — it defines the interface and wires up model
    loading, but the inference loop is not yet implemented.

    Attributes:
        model_path: Path to the ``*.pt`` weights file.
        confidence: Minimum detection confidence (0.0–1.0).
        class_names: Mapping from class index to label string.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence: float = 0.50,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO ``*.pt`` weights. If None the detector
                is in "unavailable" mode and all find calls return empty.
            confidence: Minimum detection score to accept a result.

        Raises:
            ImportError: If ``ultralytics`` is not installed and model_path is given.
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence = confidence
        self._model: Any = None  # ultralytics.YOLO instance, lazy-loaded
        self.class_names: dict[int, str] = {}

        if self.model_path:
            self._load_model()

    @property
    def available(self) -> bool:
        """True if a model is loaded and inference is possible."""
        return self._model is not None

    def _load_model(self) -> None:
        """Load the YOLO model from disk.

        Raises:
            ImportError: If ``ultralytics`` is not installed.
            FileNotFoundError: If the weights file does not exist.
        """
        # TODO: from ultralytics import YOLO; self._model = YOLO(self.model_path)
        #       self.class_names = self._model.names
        raise NotImplementedError("TODO: implement YOLODetector._load_model")

    def find(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Detect the first UI element matching *query* by class label.

        Args:
            image: Screenshot to run inference on.
            query: Class label to search for (e.g. ``"button"``, ``"text_field"``).
            threshold: Override instance confidence for this call.
            region: Optional bounding box restricting the search area.

        Returns:
            :class:`FindResult` for the highest-confidence matching detection,
            or None if no match above threshold.
        """
        if not self.available:
            return None
        # TODO: run inference, filter by class label matching query,
        #       return best result above threshold
        raise NotImplementedError("TODO: implement YOLODetector.find")

    def find_all(
        self,
        image: Image.Image,
        query: str,
        threshold: float | None = None,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Detect all UI elements matching *query* by class label.

        Args:
            image: Screenshot to run inference on.
            query: Class label to find.
            threshold: Override confidence threshold for this call.
            region: Optional search region.

        Returns:
            List of :class:`FindResult` sorted by confidence descending.
        """
        if not self.available:
            return []
        # TODO: run inference, filter, sort, return list
        raise NotImplementedError("TODO: implement YOLODetector.find_all")

    def detect_all(self, image: Image.Image) -> list[FindResult]:
        """Detect ALL UI elements visible in *image* regardless of class.

        Useful for screen-state identification and exploratory automation.

        Args:
            image: Screenshot to analyse.

        Returns:
            List of all detected elements as :class:`FindResult` objects.
        """
        if not self.available:
            return []
        # TODO: run inference, convert all boxes to FindResult
        raise NotImplementedError("TODO: implement YOLODetector.detect_all")
