"""OCR module with a pluggable engine abstraction.

The default engine is Tesseract (via ``pytesseract``).
PaddleOCR is available as an optional alternative by setting
``config.vision.ocr_engine = "paddleocr"``.

All engines implement the :class:`OCREngine` Protocol and return
:class:`~screenwalker.vision.screen_state.FindResult` objects so that the rest
of the system is engine-agnostic.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from screenwalker.vision.screen_state import BBox, FindResult


class OCREngine(Protocol):
    """Protocol for OCR engine implementations."""

    def extract_text(self, image: Image.Image) -> str:
        """Extract all text from *image* as a single string.

        Args:
            image: Input PIL image (may be pre-processed).

        Returns:
            Full extracted text, tokens joined by spaces.
        """
        ...

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Locate *query* text within *image*.

        Args:
            image: Screenshot to search.
            query: Text string to find.
            threshold: Minimum fuzzy-match confidence to accept.
            region: Optional bounding box restricting the search area.

        Returns:
            :class:`~screenwalker.vision.screen_state.FindResult` or None.
        """
        ...

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Find all occurrences of *query* in *image*.

        Args:
            image: Screenshot to search.
            query: Text string to find.
            threshold: Minimum confidence.
            region: Optional search region.

        Returns:
            List of :class:`FindResult` sorted by confidence descending.
        """
        ...


class TesseractEngine:
    """OCR engine backed by Tesseract via pytesseract.

    Attributes:
        lang: Tesseract language code(s) (e.g. ``"eng"`` or ``"eng+rus"``).
        config: Additional Tesseract config string (e.g. ``"--psm 6"``).
        preprocess: Apply grayscale + binarization before OCR when True.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 6",
        preprocess: bool = True,
    ) -> None:
        """Initialize TesseractEngine.

        Args:
            lang: Tesseract language code(s).
            config: Tesseract CLI config flags.
            preprocess: Whether to apply image pre-processing.
        """
        self.lang = lang
        self.config = config
        self.preprocess = preprocess

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Convert to grayscale and apply adaptive threshold.

        Args:
            image: Input PIL image.

        Returns:
            Pre-processed PIL image.
        """
        # TODO: convert to grayscale, apply OpenCV adaptive threshold
        raise NotImplementedError("TODO: implement OCR pre-processing")

    def extract_text(self, image: Image.Image) -> str:
        """Extract all text from *image*.

        Args:
            image: Input PIL image.

        Returns:
            Extracted text string.
        """
        # TODO: import pytesseract; return pytesseract.image_to_string(processed)
        raise NotImplementedError("TODO: implement TesseractEngine.extract_text")

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Find *query* text in *image* using Tesseract word-level data.

        Args:
            image: Screenshot to search.
            query: Text to find.
            threshold: Minimum fuzzy match score (0.0–1.0).
            region: Optional search region.

        Returns:
            Best matching :class:`FindResult` or None.
        """
        # TODO: call pytesseract.image_to_data, fuzzy-match each word/phrase,
        #       return FindResult for highest scoring match above threshold
        raise NotImplementedError("TODO: implement TesseractEngine.find_text")

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Find all occurrences of *query* in *image*.

        Args:
            image: Screenshot to search.
            query: Text to find.
            threshold: Minimum confidence.
            region: Optional search region.

        Returns:
            List of :class:`FindResult` sorted by confidence descending.
        """
        # TODO: same as find_text but collect all matches
        raise NotImplementedError("TODO: implement TesseractEngine.find_all_text")


class PaddleOCREngine:
    """Optional OCR engine backed by PaddleOCR.

    Requires the ``paddleocr`` extras: ``pip install screenwalker[paddleocr]``.

    Attributes:
        lang: PaddleOCR language code (e.g. ``"en"``, ``"ch"``).
    """

    def __init__(self, lang: str = "en") -> None:
        """Initialize PaddleOCREngine.

        Args:
            lang: PaddleOCR language code.

        Raises:
            ImportError: If paddleocr is not installed.
        """
        # TODO: try importing paddleocr; raise ImportError with install hint if missing
        self.lang = lang
        raise NotImplementedError("TODO: implement PaddleOCREngine.__init__")

    def extract_text(self, image: Image.Image) -> str:
        """Extract all text from *image*.

        Args:
            image: Input PIL image.

        Returns:
            Extracted text string.
        """
        raise NotImplementedError("TODO: implement PaddleOCREngine.extract_text")

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> FindResult | None:
        """Find *query* in *image* using PaddleOCR bounding boxes.

        Args:
            image: Screenshot to search.
            query: Text to find.
            threshold: Minimum confidence.
            region: Optional search region.

        Returns:
            Best matching :class:`FindResult` or None.
        """
        raise NotImplementedError("TODO: implement PaddleOCREngine.find_text")

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
    ) -> list[FindResult]:
        """Find all occurrences of *query* in *image*.

        Args:
            image: Screenshot to search.
            query: Text to find.
            threshold: Minimum confidence.
            region: Optional search region.

        Returns:
            List of :class:`FindResult` sorted by confidence descending.
        """
        raise NotImplementedError("TODO: implement PaddleOCREngine.find_all_text")


def build_ocr_engine(engine_name: str = "tesseract", **kwargs) -> OCREngine:
    """Factory function to create an OCR engine by name.

    Args:
        engine_name: Engine identifier — ``"tesseract"`` or ``"paddleocr"``.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An :class:`OCREngine`-compatible engine instance.

    Raises:
        ValueError: If *engine_name* is not recognised.
    """
    engines = {
        "tesseract": TesseractEngine,
        "paddleocr": PaddleOCREngine,
    }
    cls = engines.get(engine_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown OCR engine: {engine_name!r}. "
            f"Available: {list(engines)}"
        )
    return cls(**kwargs)  # type: ignore[return-value]
