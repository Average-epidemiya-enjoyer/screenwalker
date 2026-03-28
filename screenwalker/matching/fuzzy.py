"""Fuzzy text matching using RapidFuzz.

Provides normalisation helpers and a :class:`FuzzyMatcher` that scores
query strings against a corpus of OCR-extracted text tokens.

Scoring conventions
-------------------
* Module-level :func:`fuzzy_score` and :func:`fuzzy_match` use **float** scores
  in ``[0.0, 1.0]`` for backward compatibility with the vision pipeline.
* :class:`FuzzyMatcher` new-style methods (:meth:`~FuzzyMatcher.match`,
  :meth:`~FuzzyMatcher.best_match`, :meth:`~FuzzyMatcher.contains_fuzzy`) use
  **int** scores in ``[0, 100]`` to match the RapidFuzz convention directly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz as _fuzz
from rapidfuzz import process as _process


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzyResult:
    """A single fuzzy-match result.

    Attributes:
        text: The original candidate string (before any normalisation).
        score: RapidFuzz similarity score in ``[0, 100]``.
        original_index: Position of the candidate in the input list.
    """

    text: str
    score: int
    original_index: int


# ---------------------------------------------------------------------------
# String normalisation
# ---------------------------------------------------------------------------


def normalise(text: str, lowercase: bool = True, strip_whitespace: bool = True) -> str:
    """Normalise a string for fuzzy comparison.

    Applies Unicode NFC normalisation, optional lowercasing, whitespace
    collapsing, and removal of common non-semantic characters.

    Args:
        text: Raw input string.
        lowercase: Convert to lowercase when True.
        strip_whitespace: Collapse runs of whitespace to single spaces.

    Returns:
        Normalised string.

    Example:
        >>> normalise("  Hello\\tWorld!  ")
        'hello world!'
    """
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# Alias with English spelling so both names are importable
normalize = normalise


# ---------------------------------------------------------------------------
# Module-level convenience functions (float 0-1 scores, backward-compat)
# ---------------------------------------------------------------------------


def fuzzy_score(query: str, candidate: str, normalise_inputs: bool = True) -> float:
    """Compute a fuzzy similarity score between *query* and *candidate*.

    Uses ``rapidfuzz.fuzz.WRatio`` (weighted ratio) which combines partial
    ratio, token sort ratio, and token set ratio for best real-world results.

    Args:
        query: The search string (e.g. from the scenario YAML).
        candidate: The candidate string (e.g. from OCR output).
        normalise_inputs: Apply :func:`normalise` to both strings before scoring.

    Returns:
        Score in the range [0.0, 1.0].
    """
    q = normalise(query) if normalise_inputs else query
    c = normalise(candidate) if normalise_inputs else candidate
    return _fuzz.WRatio(q, c) / 100.0


def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.80,
    normalise_inputs: bool = True,
) -> list[tuple[str, float]]:
    """Find all candidates that fuzzy-match *query* above *threshold*.

    Args:
        query: Search string.
        candidates: List of strings to compare against.
        threshold: Minimum score (0.0–1.0) to include in results.
        normalise_inputs: Normalise strings before scoring.

    Returns:
        List of ``(candidate, score)`` tuples sorted by score descending,
        filtered to entries where score >= threshold.
    """
    if not candidates:
        return []

    q = normalise(query) if normalise_inputs else query
    normed = [normalise(c) if normalise_inputs else c for c in candidates]

    cutoff = int(threshold * 100)
    raw = _process.extract(
        q,
        normed,
        scorer=_fuzz.WRatio,
        score_cutoff=cutoff,
        limit=None,
    )
    # raw items: (matched_text, score, index)
    return sorted(
        [(candidates[idx], score / 100.0) for _, score, idx in raw],
        key=lambda t: t[1],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# FuzzyMatcher class
# ---------------------------------------------------------------------------


class FuzzyMatcher:
    """Stateful fuzzy matcher with configurable thresholds.

    New-style methods (:meth:`match`, :meth:`best_match`, :meth:`contains_fuzzy`)
    use **int** scores 0–100. Legacy methods (:meth:`score`, :meth:`all_matches`)
    use float scores 0.0–1.0 for backward compatibility with the vision pipeline.

    Attributes:
        threshold: Default minimum score for new-style methods (0–100).
        normalise_inputs: Whether to normalise strings before scoring.
    """

    def __init__(self, threshold: int = 80, normalise_inputs: bool = True) -> None:
        """Initialize FuzzyMatcher.

        Args:
            threshold: Default score threshold (0–100) for :meth:`match` and
                :meth:`best_match`.
            normalise_inputs: Apply string normalisation before scoring.
        """
        self.threshold = threshold
        self.normalise_inputs = normalise_inputs

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(text: str) -> str:
        """Normalise *text* for comparison (lowercase, collapse whitespace).

        Args:
            text: Raw input string.

        Returns:
            Normalised string.
        """
        return normalise(text)

    # ------------------------------------------------------------------
    # New-style methods — int scores 0-100
    # ------------------------------------------------------------------

    def match(
        self,
        query: str,
        candidates: list[str],
        threshold: int | None = None,
    ) -> list[FuzzyResult]:
        """Find all candidates that fuzzy-match *query* above *threshold*.

        Uses ``rapidfuzz.fuzz.token_sort_ratio`` for the primary score and
        ``rapidfuzz.process.extract`` for efficient batch processing.

        Args:
            query: Search string.
            candidates: Pool of candidate strings.
            threshold: Minimum score 0–100 (defaults to :attr:`threshold`).

        Returns:
            List of :class:`FuzzyResult` sorted by score descending.
        """
        if not candidates:
            return []

        cutoff = threshold if threshold is not None else self.threshold
        q = normalise(query) if self.normalise_inputs else query
        normed = [normalise(c) if self.normalise_inputs else c for c in candidates]

        raw = _process.extract(
            q,
            normed,
            scorer=_fuzz.token_sort_ratio,
            score_cutoff=cutoff,
            limit=None,
        )
        results = [
            FuzzyResult(text=candidates[idx], score=int(score), original_index=idx)
            for _, score, idx in raw
        ]
        return sorted(results, key=lambda r: r.score, reverse=True)

    def best_match(
        self,
        query: str,
        candidates: list[str],
        threshold: int | None = None,
    ) -> FuzzyResult | None:
        """Return the best matching candidate above *threshold*.

        Args:
            query: Search string.
            candidates: Pool of candidate strings.
            threshold: Minimum score 0–100 (defaults to :attr:`threshold`).

        Returns:
            :class:`FuzzyResult` for the best match, or None if nothing
            exceeds the threshold.
        """
        if not candidates:
            return None

        cutoff = threshold if threshold is not None else self.threshold
        q = normalise(query) if self.normalise_inputs else query
        normed = {i: (normalise(c) if self.normalise_inputs else c) for i, c in enumerate(candidates)}

        result = _process.extractOne(
            q,
            normed,
            scorer=_fuzz.token_sort_ratio,
            score_cutoff=cutoff,
        )
        if result is None:
            return None
        _matched_text, score, idx = result
        return FuzzyResult(text=candidates[idx], score=int(score), original_index=idx)

    def contains_fuzzy(
        self,
        text: str,
        substring: str,
        threshold: int = 85,
    ) -> bool:
        """Check whether *substring* appears inside *text* with fuzzy matching.

        Uses ``rapidfuzz.fuzz.partial_ratio`` which finds the best alignment
        of the shorter string within the longer one.

        Args:
            text: The longer string to search within.
            substring: The shorter string to search for.
            threshold: Minimum ``partial_ratio`` score (0–100).

        Returns:
            True when the best partial match score >= *threshold*.
        """
        t = normalise(text) if self.normalise_inputs else text
        s = normalise(substring) if self.normalise_inputs else substring
        return int(_fuzz.partial_ratio(s, t)) >= threshold

    # ------------------------------------------------------------------
    # Legacy methods — float scores 0.0-1.0
    # ------------------------------------------------------------------

    def score(self, query: str, candidate: str) -> float:
        """Score *query* against a single *candidate*.

        Args:
            query: Search string.
            candidate: Candidate to score.

        Returns:
            Similarity score in [0.0, 1.0] (WRatio).
        """
        return fuzzy_score(query, candidate, normalise_inputs=self.normalise_inputs)

    def all_matches(
        self, query: str, candidates: list[str], threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """Return all candidates matching *query* above threshold.

        Args:
            query: Search string.
            candidates: Pool of candidate strings.
            threshold: Override instance threshold for this call (0.0–1.0).

        Returns:
            Sorted list of ``(candidate, score)`` tuples, scores in [0.0, 1.0].
        """
        t = threshold if threshold is not None else self.threshold / 100.0
        return fuzzy_match(
            query, candidates, threshold=t, normalise_inputs=self.normalise_inputs
        )
