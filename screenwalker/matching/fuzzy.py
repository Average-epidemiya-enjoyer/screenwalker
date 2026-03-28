"""Fuzzy text matching using RapidFuzz.

Provides normalisation helpers and a :class:`FuzzyMatcher` that scores
query strings against a corpus of OCR-extracted text tokens.
"""

from __future__ import annotations

import re
import unicodedata


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
    # TODO: from rapidfuzz import fuzz
    #       q = normalise(query) if normalise_inputs else query
    #       c = normalise(candidate) if normalise_inputs else candidate
    #       return fuzz.WRatio(q, c) / 100.0
    raise NotImplementedError("TODO: implement fuzzy_score")


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
    # TODO: rapidfuzz.process.extract(query, candidates, scorer=fuzz.WRatio, ...)
    raise NotImplementedError("TODO: implement fuzzy_match")


class FuzzyMatcher:
    """Stateful fuzzy matcher with configurable thresholds.

    Attributes:
        threshold: Default minimum score (0.0–1.0).
        normalise: Whether to normalise strings before scoring.
    """

    def __init__(self, threshold: float = 0.80, normalise: bool = True) -> None:
        """Initialize FuzzyMatcher.

        Args:
            threshold: Default score threshold for matches.
            normalise: Apply string normalisation before scoring.
        """
        self.threshold = threshold
        self.normalise = normalise

    def score(self, query: str, candidate: str) -> float:
        """Score *query* against a single *candidate*.

        Args:
            query: Search string.
            candidate: Candidate to score.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        return fuzzy_score(query, candidate, normalise_inputs=self.normalise)

    def best_match(
        self, query: str, candidates: list[str]
    ) -> tuple[str, float] | None:
        """Return the best matching candidate above threshold.

        Args:
            query: Search string.
            candidates: Pool of candidate strings.

        Returns:
            ``(best_candidate, score)`` or None if no match above threshold.
        """
        # TODO: call fuzzy_match and return first result
        raise NotImplementedError("TODO: implement FuzzyMatcher.best_match")

    def all_matches(
        self, query: str, candidates: list[str], threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """Return all candidates matching *query* above threshold.

        Args:
            query: Search string.
            candidates: Pool of candidate strings.
            threshold: Override instance threshold for this call.

        Returns:
            Sorted list of ``(candidate, score)`` tuples.
        """
        t = threshold if threshold is not None else self.threshold
        return fuzzy_match(query, candidates, threshold=t, normalise_inputs=self.normalise)
