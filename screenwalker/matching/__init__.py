"""Text matching utilities — fuzzy matching and UI synonym resolution."""

from screenwalker.matching.fuzzy import FuzzyMatcher, fuzzy_match, fuzzy_score
from screenwalker.matching.synonyms import SynonymRegistry

__all__ = ["FuzzyMatcher", "fuzzy_match", "fuzzy_score", "SynonymRegistry"]
