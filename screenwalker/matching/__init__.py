"""Text matching utilities — fuzzy matching and UI synonym resolution."""

from screenwalker.matching.fuzzy import FuzzyMatcher, FuzzyResult, fuzzy_match, fuzzy_score
from screenwalker.matching.synonyms import SynonymDictionary, SynonymRegistry

__all__ = [
    "FuzzyMatcher",
    "FuzzyResult",
    "fuzzy_match",
    "fuzzy_score",
    "SynonymDictionary",
    "SynonymRegistry",
]
