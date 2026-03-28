"""OCR module with a pluggable engine abstraction.

The default engine is Tesseract (via ``pytesseract``).
PaddleOCR is available as an optional alternative by setting
``config.vision.ocr_engine = "paddleocr"``.

All engines implement the :class:`OCREngine` Protocol.  Word-level OCR
output is represented by :class:`OCRResult`; the higher-level
:class:`~screenwalker.vision.screen_state.FindResult` is returned by the
``find_text`` / ``find_all_text`` finders so the rest of the system stays
engine-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PIL import Image
from rapidfuzz import fuzz as _rfuzz

from screenwalker.vision.screen_state import BBox, FindResult

if TYPE_CHECKING:
    from screenwalker.utils.config import AppConfig

# ---------------------------------------------------------------------------
# Optional heavy dependencies — fail at *call* time, not at import time,
# so the module is importable even without the binaries installed.
# The module-level names are patchable in unit tests.
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# OCR result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRResult:
    """Word- or line-level OCR output.

    Attributes:
        text: Recognised text string.
        confidence: Normalised confidence score in [0.0, 1.0].
        bbox: Bounding box of the text in image coordinates.
        raw_data: Engine-specific metadata (Tesseract level/block/line/word,
            PaddleOCR polygon, etc.).
    """

    text: str
    confidence: float
    bbox: BBox
    raw_data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol (structural interface)
# ---------------------------------------------------------------------------


class OCREngine(Protocol):
    """Structural protocol for OCR engine implementations."""

    def recognize(
        self, image: Image.Image, lang: str = "rus+eng"
    ) -> list[OCRResult]:
        """Run OCR on *image* and return word-level results.

        Args:
            image: Input PIL image.
            lang: Language hint (engine-specific format).

        Returns:
            List of :class:`OCRResult` filtered by confidence threshold.
        """
        ...

    def extract_text(self, image: Image.Image) -> str:
        """Extract all recognised text as a single space-joined string.

        Args:
            image: Input PIL image.

        Returns:
            Extracted text string.
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
            threshold: Minimum fuzzy-match confidence in [0.0, 1.0].
            region: Optional bounding box restricting the search area.

        Returns:
            Best :class:`FindResult` above *threshold*, or None.
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


# ---------------------------------------------------------------------------
# Tesseract engine
# ---------------------------------------------------------------------------


class TesseractEngine:
    """OCR engine backed by Tesseract via pytesseract.

    Pre-processing pipeline (applied when *preprocess* is True):
    1. Convert to grayscale.
    2. Optionally double the resolution (*resize*) for small text.
    3. Apply ``cv2.adaptiveThreshold`` binarisation.
    4. Optionally denoise with ``cv2.fastNlMeansDenoising`` (*denoise*).

    Attributes:
        lang: Tesseract language code(s) (e.g. ``"eng"`` or ``"eng+rus"``).
        config: Additional Tesseract config string (e.g. ``"--psm 6"``).
        preprocess: Apply the pre-processing pipeline before OCR.
        confidence_threshold: Minimum Tesseract word confidence (0–100).
            Words below this value are discarded. Default: 60.
        resize: Double the image resolution before OCR (helps with small text).
        denoise: Apply non-local means denoising before OCR.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 6",
        preprocess: bool = True,
        confidence_threshold: int = 60,
        resize: bool = False,
        denoise: bool = False,
    ) -> None:
        self.lang = lang
        self.config = config
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.resize = resize
        self.denoise = denoise

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Apply the pre-processing pipeline to *image*.

        Args:
            image: Input PIL image (any mode).

        Returns:
            Pre-processed PIL image in ``"L"`` (grayscale) mode.

        Raises:
            ImportError: If *opencv-python-headless* is not installed.
        """
        if cv2 is None:
            raise ImportError(
                "opencv-python-headless is required for OCR pre-processing: "
                "pip install opencv-python-headless"
            )

        # Step 1 – grayscale
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Step 2 – optional ×2 resize (bicubic for quality)
        if self.resize:
            h, w = gray.shape
            gray = cv2.resize(
                gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC
            )

        # Step 3 – adaptive binarisation
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Step 4 – optional denoising
        if self.denoise:
            binary = cv2.fastNlMeansDenoising(binary, h=10)

        return Image.fromarray(binary)

    # ------------------------------------------------------------------
    # Core recognition
    # ------------------------------------------------------------------

    def recognize(
        self, image: Image.Image, lang: str | None = None
    ) -> list[OCRResult]:
        """Run Tesseract on *image* and return filtered word-level results.

        Args:
            image: Input PIL image.
            lang: Override engine language for this call.

        Returns:
            List of :class:`OCRResult` with confidence >=
            :attr:`confidence_threshold` / 100.

        Raises:
            ImportError: If *pytesseract* is not installed.
        """
        if pytesseract is None:
            raise ImportError(
                "pytesseract is required: pip install pytesseract"
            )

        effective_lang = lang if lang is not None else self.lang
        processed = self._preprocess(image) if self.preprocess else image

        data = pytesseract.image_to_data(
            processed,
            lang=effective_lang,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )

        results: list[OCRResult] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            try:
                conf = int(data["conf"][i])
            except (ValueError, TypeError):
                continue
            # conf == -1 for non-word rows (block/line/paragraph headers)
            if not text or conf < 0 or conf < self.confidence_threshold:
                continue

            bbox = BBox(
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
            )
            results.append(
                OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=bbox,
                    raw_data={
                        "level": data.get("level", [None] * n)[i],
                        "block_num": data.get("block_num", [None] * n)[i],
                        "line_num": data.get("line_num", [None] * n)[i],
                        "word_num": data.get("word_num", [None] * n)[i],
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # Line grouping
    # ------------------------------------------------------------------

    def _group_into_lines(
        self,
        results: list[OCRResult],
    ) -> list[OCRResult]:
        """Merge word-level results into line-level results.

        Two words are considered to be on the same line when the distance
        between their vertical centres is within half the height of the
        taller word.

        Args:
            results: Word-level :class:`OCRResult` list (from :meth:`recognize`).

        Returns:
            Line-level :class:`OCRResult` list sorted top-to-bottom,
            left-to-right.
        """
        if not results:
            return []

        # Sort by (y_center, x) so left-to-right within a line
        def _cy(r: OCRResult) -> int:
            return r.bbox.y + r.bbox.h // 2

        sorted_words = sorted(results, key=lambda r: (_cy(r), r.bbox.x))

        lines: list[list[OCRResult]] = []
        current_line: list[OCRResult] = [sorted_words[0]]

        for word in sorted_words[1:]:
            last = current_line[-1]
            tol = max(last.bbox.h, word.bbox.h) // 2
            if abs(_cy(word) - _cy(last)) <= tol:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        merged: list[OCRResult] = []
        for line_words in lines:
            text = " ".join(w.text for w in line_words)
            avg_conf = sum(w.confidence for w in line_words) / len(line_words)
            min_x = min(w.bbox.x for w in line_words)
            min_y = min(w.bbox.y for w in line_words)
            max_x = max(w.bbox.x + w.bbox.w for w in line_words)
            max_y = max(w.bbox.y + w.bbox.h for w in line_words)
            merged.append(
                OCRResult(
                    text=text,
                    confidence=avg_conf,
                    bbox=BBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y),
                    raw_data={},
                )
            )
        return merged

    # ------------------------------------------------------------------
    # High-level finders
    # ------------------------------------------------------------------

    def extract_text(self, image: Image.Image) -> str:
        """Extract all text from *image* as a space-joined string.

        Args:
            image: Input PIL image.

        Returns:
            Recognised text, words joined by spaces.
        """
        return " ".join(r.text for r in self.recognize(image))

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> FindResult | None:
        """Find the best-matching occurrence of *query* in *image*.

        Args:
            image: Screenshot to search.
            query: Text to find.
            threshold: Minimum match confidence in [0.0, 1.0].
            region: Optional bounding box restricting the search area.
            fuzzy: Use RapidFuzz ``partial_ratio`` when True; exact
                substring check when False.
            synonym_registry: Optional :class:`~screenwalker.matching.SynonymRegistry`
                used to expand *query* when the direct search finds nothing.

        Returns:
            :class:`FindResult` for the best match, or None.
        """
        matches = self.find_all_text(
            image, query, threshold=threshold, region=region, fuzzy=fuzzy,
            synonym_registry=synonym_registry,
        )
        return matches[0] if matches else None

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> list[FindResult]:
        """Find all occurrences of *query* in *image*.

        When *synonym_registry* is provided and the direct search returns no
        results, the query is expanded to all known aliases and each alias is
        searched in turn.

        Args:
            image: Screenshot to search.
            query: Text to find.
            threshold: Minimum confidence in [0.0, 1.0].
            region: Optional bounding box restricting the search area.
            fuzzy: Use fuzzy matching when True.
            synonym_registry: Optional registry for synonym expansion fallback.

        Returns:
            List of :class:`FindResult` sorted by confidence descending.
        """
        search_image = image
        offset_x, offset_y = 0, 0
        if region is not None:
            search_image = image.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            offset_x, offset_y = region.x, region.y

        word_results = self.recognize(search_image)
        line_results = self._group_into_lines(word_results)

        def _score_lines(q: str) -> list[FindResult]:
            q_norm = q.lower().strip()
            found: list[FindResult] = []
            for result in line_results:
                text_norm = result.text.lower().strip()
                if not text_norm:
                    continue
                if fuzzy:
                    score = _rfuzz.partial_ratio(q_norm, text_norm) / 100.0
                else:
                    score = 1.0 if q_norm in text_norm else 0.0
                if score >= threshold:
                    final_bbox = result.bbox.offset(offset_x, offset_y)
                    found.append(
                        FindResult(
                            element=result.text,
                            confidence=score,
                            bbox=final_bbox,
                            method="ocr",
                            metadata={"ocr_confidence": result.confidence},
                        )
                    )
            return found

        matches = _score_lines(query)

        if not matches and synonym_registry is not None:
            aliases = synonym_registry.all_aliases(query) - {query.lower().strip()}
            for alias in aliases:
                alias_matches = _score_lines(alias)
                matches.extend(alias_matches)

        return sorted(matches, key=lambda r: r.confidence, reverse=True)


# ---------------------------------------------------------------------------
# PaddleOCR engine (optional)
# ---------------------------------------------------------------------------


class PaddleOCREngine:
    """Optional OCR engine backed by PaddleOCR.

    Requires the ``paddleocr`` extras: ``pip install screenwalker[paddleocr]``.

    Attributes:
        lang: PaddleOCR language code (e.g. ``"ru"``, ``"en"``, ``"ch"``).
    """

    def __init__(self, lang: str = "ru") -> None:
        """Initialise PaddleOCREngine.

        Args:
            lang: PaddleOCR language code.

        Raises:
            ImportError: If paddleocr is not installed.
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install with: pip install screenwalker[paddleocr]"
            ) from exc

        self.lang = lang
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def recognize(
        self, image: Image.Image, lang: str | None = None
    ) -> list[OCRResult]:
        """Run PaddleOCR on *image*.

        Args:
            image: Input PIL image.
            lang: Ignored — PaddleOCR language is set at construction time.

        Returns:
            List of :class:`OCRResult` one per detected text box.
        """
        img_array = np.array(image.convert("RGB"))
        raw = self._ocr.ocr(img_array, cls=True)

        results: list[OCRResult] = []
        # raw is [[line, ...]] where line = [box_points, (text, confidence)]
        page = raw[0] if raw else []
        for item in page:
            box, (text, conf) = item
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs)) - x
            h = int(max(ys)) - y
            results.append(
                OCRResult(
                    text=str(text),
                    confidence=float(conf),
                    bbox=BBox(x=x, y=y, w=w, h=h),
                    raw_data={"box": box},
                )
            )
        return results

    def extract_text(self, image: Image.Image) -> str:
        """Extract all text from *image* as a space-joined string."""
        return " ".join(r.text for r in self.recognize(image))

    def find_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> FindResult | None:
        """Find the best-matching occurrence of *query* in *image*."""
        matches = self.find_all_text(
            image, query, threshold=threshold, region=region, fuzzy=fuzzy,
            synonym_registry=synonym_registry,
        )
        return matches[0] if matches else None

    def find_all_text(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.70,
        region: BBox | None = None,
        fuzzy: bool = True,
        synonym_registry: Any | None = None,
    ) -> list[FindResult]:
        """Find all occurrences of *query* in *image*."""
        search_image = image
        offset_x, offset_y = 0, 0
        if region is not None:
            search_image = image.crop(
                (region.x, region.y, region.right, region.bottom)
            )
            offset_x, offset_y = region.x, region.y

        ocr_results = self.recognize(search_image)

        def _score_results(q: str) -> list[FindResult]:
            q_norm = q.lower().strip()
            found: list[FindResult] = []
            for result in ocr_results:
                text_norm = result.text.lower().strip()
                if not text_norm:
                    continue
                if fuzzy:
                    score = _rfuzz.partial_ratio(q_norm, text_norm) / 100.0
                else:
                    score = 1.0 if q_norm in text_norm else 0.0
                if score >= threshold:
                    final_bbox = result.bbox.offset(offset_x, offset_y)
                    found.append(
                        FindResult(
                            element=result.text,
                            confidence=score,
                            bbox=final_bbox,
                            method="ocr",
                            metadata={"ocr_confidence": result.confidence},
                        )
                    )
            return found

        matches = _score_results(query)

        if not matches and synonym_registry is not None:
            aliases = synonym_registry.all_aliases(query) - {query.lower().strip()}
            for alias in aliases:
                alias_matches = _score_results(alias)
                matches.extend(alias_matches)

        return sorted(matches, key=lambda r: r.confidence, reverse=True)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_ocr_engine(config: AppConfig) -> OCREngine:
    """Create an OCR engine from :class:`~screenwalker.utils.config.AppConfig`.

    Args:
        config: Application configuration.

    Returns:
        Configured :class:`OCREngine` instance.

    Raises:
        ValueError: If ``config.vision.ocr_engine`` is not recognised.
    """
    name = config.vision.ocr_engine
    if name == "tesseract":
        return TesseractEngine(
            lang=config.vision.ocr_lang,
            config=config.vision.ocr_config,
            preprocess=config.vision.ocr_preprocess,
        )
    if name == "paddleocr":
        return PaddleOCREngine(lang=config.vision.ocr_lang)
    raise ValueError(
        f"Unknown OCR engine: {name!r}. Available: ['tesseract', 'paddleocr']"
    )


def build_ocr_engine(engine_name: str = "tesseract", **kwargs: object) -> OCREngine:
    """Factory that creates an OCR engine by name.

    This is the low-level factory; prefer :func:`create_ocr_engine` when you
    have an :class:`~screenwalker.utils.config.AppConfig` available.

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
